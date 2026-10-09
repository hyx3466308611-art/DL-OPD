#!/usr/bin/env python3
"""Outer/inner DL-OPD scheduler (plan §10).

Each outer cycle:
  1) build/update error profile on fixed diagnostic set
  2) generate + verify hard prompts → refresh Hard Buffer
  3) run inner Mask-Policy steps (delegates to train_mask_opd for full GPU loop,
     or a lightweight subsampled inner loop when --dry-run-policy)

MVP-B focuses on rule resampling + rewrite/minimal-pair operators.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dl_opd.data_policy.data_value import concept_sampling_probs, sampling_entropy
from dl_opd.data_policy.error_profile import (
    CONCEPT_KEYS,
    DiagnosticRecord,
    compute_error_profile,
    dominant_error_dims,
)
from dl_opd.data_policy.hard_ops import GeneratorInput, generate_candidates, select_operators
from dl_opd.data_policy.verify_qdata import verify_candidate
from dl_opd.hard_buffer import HardBuffer, HardSample, new_sample_id


def _load_yaml(path: str) -> dict:
    p = Path(path)
    # prefer sibling .json when PyYAML missing
    json_sidecar = p.with_suffix(".json")
    try:
        import yaml

        if p.is_file():
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except ImportError:
        pass
    if json_sidecar.is_file():
        print(f"[cfg] loading {json_sidecar} (no PyYAML)")
        return json.loads(json_sidecar.read_text(encoding="utf-8"))
    if p.is_file() and p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    print("[warn] no config loaded; using defaults")
    return {}


def load_diag_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def bootstrap_profile_from_manifest(
    rows: list[dict],
    *,
    outer_cycle: int,
    fake_iou_key: str = "iou",
) -> tuple:
    """Build DiagnosticRecords from a diag manifest.

    If predictions are absent, uses placeholder IoU=0.5 so the pipeline runs;
    real runs should call build_error_profile.py after student eval.
    """
    recs = []
    for i, r in enumerate(rows):
        concept = r.get("concept", "entities")
        if concept not in CONCEPT_KEYS:
            # normalize common labels
            concept = {
                "entity": "entities",
                "spatial & layout": "spatial",
                "relations & events": "relations",
                "affordances & functions": "affordances",
                "physics & safety": "physics",
            }.get(str(concept).lower(), "entities")
        gt_empty = bool(r.get("gt_empty", r.get("is_negative", False)))
        iou = float(r.get(fake_iou_key, 0.5))
        pred_empty = bool(r.get("pred_empty", False))
        recs.append(
            DiagnosticRecord(
                sample_id=str(r.get("id", i)),
                image_id=str(r.get("image", r.get("image_id", i))),
                prompt=str(r.get("prompt", "")),
                concept=concept,
                is_negative=gt_empty,
                iou=iou,
                pred_empty=pred_empty,
                gt_empty=gt_empty,
                confidence=r.get("confidence"),
                boundary_f=r.get("boundary_f"),
            )
        )
    profile = compute_error_profile(recs, outer_cycle=outer_cycle)
    return profile, recs


def run_outer_cycle(args, cfg: dict, cycle: int, buffer: HardBuffer, rng: random.Random) -> dict:
    data_cfg = cfg.get("data_policy", {})
    diag_path = Path(args.diag_jsonl or data_cfg.get("diag_jsonl", "data/dl_opd/diag_manifest.jsonl"))
    if not diag_path.is_file():
        # synthesize minimal diag from phase2 head for dry-run
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        src = ROOT / "data" / "phase2" / "phase2_mixed.jsonl"
        rows = []
        if src.is_file():
            with src.open("r", encoding="utf-8") as f:
                for j, line in enumerate(f):
                    if j >= int(data_cfg.get("diag_bootstrap_n", 64)):
                        break
                    item = json.loads(line)
                    rows.append(
                        {
                            "id": item.get("id", j),
                            "image": item.get("image"),
                            "prompt": item.get("prompt", ""),
                            "concept": item.get("concept", "entities"),
                            "iou": 0.5,
                            "gt_empty": False,
                        }
                    )
        else:
            rows = [
                {
                    "id": i,
                    "image": f"img_{i}",
                    "prompt": f"object {i}",
                    "concept": CONCEPT_KEYS[i % 5],
                    "iou": 0.4 + 0.1 * (i % 3),
                    "gt_empty": i % 11 == 0,
                }
                for i in range(32)
            ]
        with diag_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[dl_opd] bootstrapped diag → {diag_path}")

    rows = load_diag_jsonl(diag_path, limit=args.diag_limit)
    profile, recs = bootstrap_profile_from_manifest(rows, outer_cycle=cycle)
    out_dir = Path(args.out or cfg.get("out", "ckpts/dl_opd/runs"))
    cycle_dir = out_dir / f"outer_{cycle:02d}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    profile.save(cycle_dir / "error_profile.json")

    probs = concept_sampling_probs(profile)
    (cycle_dir / "concept_probs.json").write_text(
        json.dumps({"probs": probs, "entropy": sampling_entropy(probs)}, indent=2),
        encoding="utf-8",
    )

    dom = dominant_error_dims(profile, topk=3)
    ops = select_operators(dom, max_ops=int(data_cfg.get("max_ops", 2)))
    # MVP-B: force include rewrite + minimal pair
    for must in ("semantic_rewrite", "minimal_semantic_pair"):
        if must not in ops:
            ops.append(must)

    # sample parents weighted by concept weakness
    parents = []
    for r in recs:
        if r.gt_empty:
            continue
        parents.append(
            GeneratorInput(
                image_id=r.image_id,
                image_path=str(r.meta.get("image_path", r.image_id)),
                original_prompt=r.prompt,
                concept=r.concept,
                error_type=dom[0] if dom else "E_ent",
                meta={"sample_id": r.sample_id},
            )
        )
    rng.shuffle(parents)
    parents = parents[: int(data_cfg.get("gen_parents", 32))]
    cands = generate_candidates(parents, ops, rng, max_per_image=int(data_cfg.get("max_per_image", 2)))

    accepted = rejected = 0
    for c in cands:
        q = verify_candidate(
            is_negative=c.is_negative,
            label_policy=c.expected_label_policy,
            novelty=0.7,
            duplicate=False,
        )
        payload = {
            "prompt": c.prompt,
            "operator": c.operator,
            "parent": c.parent_prompt,
            "q": q.to_dict(),
        }
        if not q.accept:
            buffer.reject(payload, q.reject_reason or "rejected")
            rejected += 1
            continue
        hs = HardSample(
            sample_id=new_sample_id(),
            image_id=c.meta.get("image_id", "unknown"),
            image_path="",
            prompt=c.prompt,
            parent_sample_id=None,
            concept=c.concept,
            error_type=c.diff_field or "gen",
            operator=c.operator,
            is_negative=c.is_negative,
            label_type="empty" if c.is_negative else "hard",
            q_data=q.q_data,
            c_sem=q.c_sem,
            c_geo=q.c_geo,
            c_teacher=q.c_teacher,
            c_novel=q.c_novel,
            value=float(profile.u.get(dom[0], 0.5)) if dom else 0.5,
            outer_cycle_created=cycle,
            meta={"parent_prompt": c.parent_prompt, "label_policy": c.expected_label_policy},
        )
        # attach image id from parent when possible
        for p in parents:
            if p.original_prompt == c.parent_prompt:
                hs.image_id = p.image_id
                hs.image_path = p.image_path
                hs.parent_sample_id = str(p.meta.get("sample_id"))
                break
        buffer.add(hs)
        accepted += 1

    ev = buffer.refresh(cycle)
    buffer.save_manifest(cycle_dir / "hard_buffer.jsonl")
    buffer.save_rejects(cycle_dir / "rejects.jsonl")
    summary = {
        "cycle": cycle,
        "u": profile.u,
        "dominant": dom,
        "operators": ops,
        "accepted": accepted,
        "rejected": rejected,
        "buffer": {"size": len(buffer), **ev, **buffer.age_stats(cycle), "concepts": buffer.concept_histogram()},
        "concept_probs": probs,
        "sampling_entropy": sampling_entropy(probs),
    }
    (cycle_dir / "outer_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(ROOT / "configs/dl_opd/dl_opd.yaml"))
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--diag_jsonl", type=str, default=None)
    ap.add_argument("--diag_limit", type=int, default=None)
    ap.add_argument("--n_outer", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--run_inner", action="store_true", help="After each outer cycle, launch train_mask_opd")
    ap.add_argument("--inner_extra_args", type=str, default="", help="Extra CLI args for train_mask_opd")
    ap.add_argument("--policy_only", action="store_true", help="Only run Data-Policy outer loops")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    n_outer = int(args.n_outer or cfg.get("schedule", {}).get("n_outer", 4))
    rng = random.Random(args.seed)
    buffer = HardBuffer(ttl_cycles=int(cfg.get("data_policy", {}).get("ttl_cycles", 2)))

    for c in range(1, n_outer + 1):
        print(f"\n===== OUTER CYCLE {c}/{n_outer} =====")
        run_outer_cycle(args, cfg, c, buffer, rng)
        if args.run_inner and not args.policy_only:
            cmd = [
                sys.executable,
                "-u",
                str(ROOT / "scripts/dl_opd/train_mask_opd.py"),
                "--config",
                args.config,
            ]
            if args.inner_extra_args:
                cmd.extend(args.inner_extra_args.split())
            print("[dl_opd] launching inner:", " ".join(cmd))
            subprocess.check_call(cmd, cwd=str(ROOT))

    print("[dl_opd] all outer cycles done")


if __name__ == "__main__":
    main()
