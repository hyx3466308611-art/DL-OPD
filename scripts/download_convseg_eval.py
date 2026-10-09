#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download aadarsh99/ConverSeg eval benchmark (Python 3.6+).

Run on server (campus net cannot reach huggingface.co — use hf-mirror):
  conda activate converseg
  export HF_ENDPOINT=https://hf-mirror.com
  python scripts/download_convseg_eval.py --out data/ConverSeg/data \\
      --workers 4 --passes 10 --retries 5 --timeout 300 --no-ssl-verify
"""

import argparse
import json
import os
import random
import ssl
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if sys.version_info < (3, 6):
    sys.exit("Need Python 3.6+. Run: conda activate converseg  then  python ...")

REPO = "aadarsh99/ConverSeg"
DEFAULT_MIRROR = "https://hf-mirror.com"
# Campus servers often cannot reach huggingface.co; try mirrors in order.
MIRROR_CANDIDATES = (
    "https://hf-mirror.com",
    "https://huggingface.co",
)
JSONL_FILES = ("data/sam_seeded.jsonl", "data/human_annotated.jsonl")
DEFAULT_TIMEOUT = 300

_print_lock = threading.Lock()
_rate_lock = threading.Lock()
_last_request = 0.0


def parse_args():
    p = argparse.ArgumentParser(description="Download ConverSeg eval dataset locally.")
    p.add_argument("--out", default="data/ConverSeg/data")
    p.add_argument("--base-url", default=os.environ.get("HF_ENDPOINT", DEFAULT_MIRROR))
    p.add_argument("--retries", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--passes", type=int, default=5)
    p.add_argument("--delay-ms", type=int, default=80)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout seconds.")
    p.add_argument(
        "--auto-mirror",
        action="store_true",
        default=True,
        help="If base-url fails, try other mirrors (default: on).",
    )
    p.add_argument("--no-auto-mirror", action="store_false", dest="auto_mirror")
    p.add_argument("--no-ssl-verify", action="store_true")
    p.add_argument("--jsonl-only", action="store_true")
    return p.parse_args()


def resolve_url(base_url, repo, relpath):
    return "{}/datasets/{}/resolve/main/{}".format(
        base_url.rstrip("/"), repo, relpath
    )


def _safe_unlink(path):
    try:
        path.unlink()
    except OSError:
        pass


def _throttle(delay_ms):
    global _last_request
    if delay_ms <= 0:
        return
    gap = delay_ms / 1000.0
    with _rate_lock:
        now = time.monotonic()
        wait = gap - (now - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()


def probe_base_url(base_url, ssl_context, timeout):
    """Return True if we can fetch a tiny manifest from this mirror."""
    url = resolve_url(base_url, REPO, JSONL_FILES[0])
    req = urllib.request.Request(url, headers={"User-Agent": "ConverSeg-download/1.0"})
    with urllib.request.urlopen(req, context=ssl_context, timeout=timeout) as resp:
        data = resp.read(4096)
    return len(data) > 100


def choose_base_url(preferred, ssl_context, timeout, auto_mirror):
    if auto_mirror:
        candidates = list(MIRROR_CANDIDATES)
    else:
        candidates = [preferred]
    last_err = None
    for base in candidates:
        print("Probing {} ...".format(base), flush=True)
        try:
            if probe_base_url(base, ssl_context, timeout):
                print("Using mirror: {}".format(base), flush=True)
                return base
        except Exception as e:
            last_err = e
            print("  unavailable: {}".format(e), flush=True)
    raise RuntimeError(
        "No mirror reachable. Tried: {}. Last error: {}".format(candidates, last_err)
    )


def download_file(url, dest, retries, ssl_context, delay_ms, timeout, quiet=False):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            _throttle(delay_ms)
            req = urllib.request.Request(
                url, headers={"User-Agent": "ConverSeg-download/1.0"}
            )
            with urllib.request.urlopen(req, context=ssl_context, timeout=timeout) as resp:
                data = resp.read()
            dest.write_bytes(data)
            if dest.stat().st_size == 0:
                raise RuntimeError("downloaded empty file")
            return
        except Exception as e:
            last_err = e
            _safe_unlink(dest)
            wait = min(2 ** attempt, 20) + random.uniform(0, 1)
            if not quiet and attempt <= 2:
                with _print_lock:
                    print("  retry {}/{}: {} ({})".format(attempt, retries, dest.name, e), flush=True)
            if attempt < retries:
                time.sleep(wait)
    raise RuntimeError(str(last_err) if last_err else "unknown error")


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def missing_files(out_dir, rel_paths):
    out = []
    out_dir = Path(out_dir)
    for rel in rel_paths:
        dest = out_dir / rel
        if not dest.is_file() or dest.stat().st_size == 0:
            out.append(rel)
    return out


def download_batch(rels, out_dir, base_url, retries, ssl_context, workers, delay_ms, timeout, have_before, total):
    if not rels:
        return 0, []

    ok = 0
    fail = 0
    failed = []
    batch_n = len(rels)
    t0 = time.time()
    last_log = t0
    out_dir = Path(out_dir)

    def _one(rel):
        url = resolve_url(base_url, REPO, "data/" + rel)
        download_file(url, out_dir / rel, retries, ssl_context, delay_ms, timeout, quiet=True)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_one, rel): rel for rel in rels}
        for fut in as_completed(futures):
            rel = futures[fut]
            try:
                fut.result()
                ok += 1
            except Exception as e:
                fail += 1
                failed.append((rel, str(e)))

            now = time.time()
            finished = ok + fail
            if finished == batch_n or finished % 20 == 0 or (now - last_log) >= 15:
                have = have_before + ok
                elapsed = now - t0
                rate = ok / elapsed if elapsed > 0 else 0.0
                eta_s = (batch_n - finished) / rate if rate > 0 else 0
                with _print_lock:
                    print(
                        "    ... batch {}/{} | total {}/{} ({:.1f}%) | ok {} fail {} | {:.2f} ok/s | ETA ~{:.0f}m".format(
                            finished, batch_n, have, total, 100.0 * have / total,
                            ok, fail, rate, eta_s / 60.0,
                        ),
                        flush=True,
                    )
                last_log = now

    return ok, failed


def main():
    args = parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ssl_context = None
    if args.no_ssl_verify:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        print("WARNING: SSL verification disabled.", flush=True)

    print("Python {}.{}".format(sys.version_info.major, sys.version_info.minor), flush=True)
    print("Output: {}".format(out_dir), flush=True)

    ssl_context = ssl_context or ssl.create_default_context()
    base_url = choose_base_url(
        args.base_url, ssl_context, args.timeout, args.auto_mirror
    )
    print("Base URL: {}".format(base_url), flush=True)

    for rel in JSONL_FILES:
        url = resolve_url(base_url, REPO, rel)
        dest = out_dir / Path(rel).name
        print("Downloading {} ...".format(dest.name), flush=True)
        download_file(url, dest, args.retries, ssl_context, args.delay_ms, args.timeout)
        print("  OK {} ({} bytes)".format(dest.name, dest.stat().st_size), flush=True)

    if args.jsonl_only:
        return 0

    rel_paths = set()
    for name in ("sam_seeded.jsonl", "human_annotated.jsonl"):
        for row in load_jsonl(out_dir / name):
            rel_paths.add(row["image"])
            rel_paths.add(row["mask"])

    all_rels = sorted(rel_paths)
    total = len(all_rels)
    workers = max(1, args.workers)
    t0 = time.time()

    for pass_idx in range(1, args.passes + 1):
        todo = missing_files(out_dir, all_rels)
        have = total - len(todo)
        if not todo:
            break

        print(
            "\nPass {}/{}: {} remaining ({}/{} done), workers={}, delay={}ms".format(
                pass_idx, args.passes, len(todo), have, total, workers, args.delay_ms
            ),
            flush=True,
        )
        have_before = have
        ok, failed = download_batch(
            todo, out_dir, base_url, args.retries, ssl_context,
            workers, args.delay_ms, args.timeout, have_before, total,
        )
        have = total - len(missing_files(out_dir, all_rels))
        elapsed = time.time() - t0
        print(
            "  pass {} ok={} fail={} | total {}/{} ({:.1f}%) | {:.0f}s".format(
                pass_idx, ok, len(failed), have, total, 100.0 * have / total, elapsed
            ),
            flush=True,
        )
        if failed and pass_idx < args.passes:
            workers = max(1, workers // 2)
            if pass_idx >= 2:
                args.delay_ms = min(args.delay_ms + 40, 300)

    todo = missing_files(out_dir, all_rels)
    if todo:
        print("\nStill missing {} file(s). Examples:".format(len(todo)), flush=True)
        for rel in todo[:5]:
            print("  {}".format(rel), flush=True)
        return 1

    n_sam = len(load_jsonl(out_dir / "sam_seeded.jsonl"))
    n_hum = len(load_jsonl(out_dir / "human_annotated.jsonl"))
    print("\nDone. sam_seeded={}, human_annotated={}, files={}".format(n_sam, n_hum, total), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run to resume.", flush=True)
        sys.exit(130)
