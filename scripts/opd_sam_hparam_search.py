#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot OPD Phase-1 (freeze Qwen, train SAM) hyperparameter search + final train.

Flow:
  1) Optuna TPE: n_trials x proxy_steps short train + num_eval on-policy Recovery
  2) Pick best trial by mean on_policy Gain
  3) Full final_steps train with best hparams -> final_out/ConverSeg_opd_final.*

Parent process stays CPU-only: train + eval run in subprocesses so GPU memory is released
between trials (avoids OOM on 2x V100).

Requires: pip install optuna

Campus example:
  cd /home/hanyuxi/ConverSeg-main
  mkdir -p ./ckpts/opd_sam_search
  unset CUDA_VISIBLE_DEVICES
  nohup python -u scripts/opd_sam_hparam_search.py ... > ./ckpts/opd_sam_search/run.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import optuna
except ImportError:
    optuna = None  # type: ignore

from train_opd import DEFAULT_STAGE2


def _run_cmd(cmd: list[str], root: str, log_path: str) -> None:
    print(f"[cmd] log -> {log_path}", flush=True)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=root, stdout=logf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = ""
        try:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                tail = "".join(f.readlines()[-50:])
        except OSError:
            pass
        raise RuntimeError(f"Command failed (exit {proc.returncode}). Last log:\n{tail}")


def _run_train(
    root: str,
    trial_dir: str,
    data_dir: str,
    dataset_jsonl: str,
    num_samples: int | None,
    steps: int,
    params: dict,
    campus: dict,
    *,
    no_save: bool = False,
    no_final_val: bool = True,
) -> None:
    os.makedirs(trial_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        os.path.join(root, "train_opd.py"),
        "--data-dir",
        data_dir,
        "--dataset-jsonl",
        dataset_jsonl,
        "--bootstrap-campus",
        "--freeze-student-plm",
        "--out",
        trial_dir,
        "--name",
        "ConverSeg_opd",
        "--steps",
        str(steps),
        "--save-every",
        "0",
        "--val-every",
        "0",
        "--log-every",
        "999999",
        "--lambda-kd",
        str(params["lambda_kd"]),
        "--kd-temperature",
        str(params["kd_temperature"]),
        "--bad-mask-mix",
        str(params["bad_mask_mix"]),
        "--lr",
        str(params["lr"]),
        "--sam-device",
        campus["sam_device"],
        "--plm-device",
        campus["plm_device"],
        "--teacher-plm-device",
        campus["teacher_plm_device"],
        "--checkpoint",
        campus["checkpoint"],
        "--plm-model-name",
        campus["plm_model_name"],
        "--model-cfg",
        campus["model_cfg"],
        "--max-dim",
        str(campus["max_dim"]),
        "--qwen-image-size",
        str(campus["qwen_image_size"]),
        "--mask-scale",
        str(campus["mask_scale"]),
        "--seed",
        str(campus["seed"]),
    ]
    if num_samples is not None:
        cmd.extend(["--num-samples", str(num_samples)])
    if no_save:
        cmd.append("--no-save")
    if no_final_val:
        cmd.append("--no-final-val")
    if campus.get("student_resume_sam"):
        cmd.extend(["--student-resume-sam", campus["student_resume_sam"]])
        cmd.extend(["--student-resume-plm", campus["student_resume_plm"]])
        cmd.extend(["--student-resume-lora", campus["student_resume_lora"]])
    _run_cmd(cmd, root, os.path.join(trial_dir, "train.log"))


def _eval_onpolicy_subprocess(
    root: str,
    ckpt_dir: str,
    data_dir: str,
    dataset_jsonl: str,
    num_eval: int,
    campus: dict,
    out_json: str,
) -> dict[str, float]:
    """Run eval in a fresh process so parent never holds GPU memory."""
    cmd = [
        sys.executable,
        "-u",
        os.path.join(root, "eval_opd_curve.py"),
        "--data-dir",
        data_dir,
        "--dataset-jsonl",
        dataset_jsonl,
        "--num-samples",
        str(num_eval),
        "--ckpt-dir",
        ckpt_dir,
        "--name",
        "ConverSeg_opd",
        "--modes",
        "on_policy",
        "--out-dir",
        os.path.join(ckpt_dir, "eval_proxy"),
        "--sam-device",
        campus["sam_device"],
        "--plm-device",
        campus["plm_device"],
        "--checkpoint",
        campus["checkpoint"],
        "--plm-model-name",
        campus["plm_model_name"],
        "--model-cfg",
        campus["model_cfg"],
        "--max-dim",
        str(campus["max_dim"]),
        "--qwen-image-size",
        str(campus["qwen_image_size"]),
        "--mask-scale",
        str(campus["mask_scale"]),
        "--seed",
        str(campus["seed"]),
    ]
    log_path = os.path.join(ckpt_dir, "eval_proxy.log")
    _run_cmd(cmd, root, log_path)

    csv_path = os.path.join(ckpt_dir, "eval_proxy", "opd_recovery_curve.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Eval CSV missing: {csv_path}")

    # Parse on_policy row from CSV
    import csv

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    on_rows = [r for r in rows if r.get("mode") == "on_policy"]
    if not on_rows:
        raise RuntimeError(f"No on_policy row in {csv_path}")
    r = on_rows[-1]
    metrics = {
        "mean_gain": float(r["mean_gain"]),
        "mean_text_iou": float(r["mean_text_iou"]),
        "mean_iou_after": float(r["mean_iou_after"]),
        "mean_iou_before": float(r["mean_iou_before"]),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser("OPD SAM hparam search (Optuna) + final full train")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dataset-jsonl", default="stage2_mixed.jsonl")
    ap.add_argument("--num-samples", type=int, default=5000)
    ap.add_argument("--num-eval", type=int, default=50)
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--proxy-steps", type=int, default=1500)
    ap.add_argument("--final-steps", type=int, default=15000)
    ap.add_argument("--study-dir", default="./ckpts/opd_sam_search")
    ap.add_argument("--final-out", default="./ckpts/stage2_opd_best")
    ap.add_argument("--skip-final", action="store_true")
    ap.add_argument("--checkpoint", default="/home/hanyuxi/ckpts_base/sam2_hiera_large.pt")
    ap.add_argument("--plm-model-name", default="./pretrain/Qwen2.5-VL-3B")
    ap.add_argument("--model-cfg", default="configs/sam2/sam2_hiera_l.yaml")
    ap.add_argument("--sam-device", default="cuda:0")
    ap.add_argument("--plm-device", default="cuda:1")
    ap.add_argument("--teacher-plm-device", default="cpu")
    ap.add_argument("--max-dim", type=int, default=768)
    ap.add_argument("--qwen-image-size", type=int, default=96)
    ap.add_argument("--mask-scale", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if optuna is None:
        raise SystemExit("Install optuna: pip install optuna")

    root = _ROOT
    campus = {
        "checkpoint": args.checkpoint,
        "plm_model_name": args.plm_model_name,
        "model_cfg": args.model_cfg,
        "sam_device": args.sam_device,
        "plm_device": args.plm_device,
        "teacher_plm_device": args.teacher_plm_device,
        "max_dim": args.max_dim,
        "qwen_image_size": args.qwen_image_size,
        "mask_scale": args.mask_scale,
        "seed": args.seed,
        "student_resume_sam": os.path.join(DEFAULT_STAGE2, "ConverSeg_stage2_final.torch"),
        "student_resume_plm": os.path.join(DEFAULT_STAGE2, "ConverSeg_stage2_plm_final.torch"),
        "student_resume_lora": os.path.join(DEFAULT_STAGE2, "lora_plm_adapter_final"),
    }
    os.makedirs(args.study_dir, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "lambda_kd": trial.suggest_float("lambda_kd", 1.0, 8.0, log=True),
            "bad_mask_mix": trial.suggest_float("bad_mask_mix", 0.0, 0.8),
            "lr": trial.suggest_float("lr", 3e-7, 3e-6, log=True),
            "kd_temperature": trial.suggest_float("kd_temperature", 1.0, 2.0),
        }
        trial_dir = os.path.join(args.study_dir, f"trial_{trial.number:03d}")
        if os.path.isdir(trial_dir):
            shutil.rmtree(trial_dir, ignore_errors=True)

        t0 = time.time()
        _run_train(
            root,
            trial_dir,
            args.data_dir,
            args.dataset_jsonl,
            args.num_samples,
            args.proxy_steps,
            params,
            campus,
            no_final_val=True,
        )
        metrics = _eval_onpolicy_subprocess(
            root,
            trial_dir,
            args.data_dir,
            args.dataset_jsonl,
            args.num_eval,
            campus,
            os.path.join(trial_dir, "metrics.json"),
        )
        gain = float(metrics["mean_gain"])
        text_iou = float(metrics["mean_text_iou"])
        trial.set_user_attr("mean_text_iou", text_iou)
        trial.set_user_attr("mean_iou_after", float(metrics["mean_iou_after"]))
        trial.set_user_attr("elapsed_s", time.time() - t0)
        trial.set_user_attr("ckpt_dir", trial_dir)
        print(
            f"[trial {trial.number}] gain={gain:.4f} text_iou={text_iou:.4f} params={params}",
            flush=True,
        )
        if text_iou < 0.35:
            return gain - 0.1
        return gain

    study = optuna.create_study(
        direction="maximize",
        study_name="opd_sam_phase1",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_trial
    best_params = best.params
    summary = {
        "best_trial": best.number,
        "best_on_policy_gain": best.value,
        "best_params": best_params,
        "best_ckpt_dir": best.user_attrs.get("ckpt_dir"),
        "n_trials": args.n_trials,
        "proxy_steps": args.proxy_steps,
    }
    summary_path = os.path.join(args.study_dir, "best_hparams.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBest trial #{best.number} gain={best.value:.4f}", flush=True)
    print(f"Best params: {best_params}", flush=True)
    print(f"Saved: {summary_path}", flush=True)

    if args.skip_final:
        print("Skip final train (--skip-final).", flush=True)
        return

    print(f"\n=== Final train {args.final_steps} steps -> {args.final_out} ===", flush=True)
    os.makedirs(args.final_out, exist_ok=True)
    _run_train(
        root,
        args.final_out,
        args.data_dir,
        args.dataset_jsonl,
        None,
        args.final_steps,
        best_params,
        campus,
        no_final_val=True,
    )
    final_metrics = _eval_onpolicy_subprocess(
        root,
        args.final_out,
        args.data_dir,
        args.dataset_jsonl,
        max(args.num_eval, 100),
        campus,
        os.path.join(args.final_out, "metrics.json"),
    )
    summary["final_on_policy_gain"] = final_metrics["mean_gain"]
    summary["final_out"] = args.final_out
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Done. Final weights: {args.final_out}/ConverSeg_opd_final.torch", flush=True)
    print(f"Final on_policy gain: {final_metrics['mean_gain']:.4f}", flush=True)


if __name__ == "__main__":
    main()
