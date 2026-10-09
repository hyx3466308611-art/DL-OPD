#!/usr/bin/env python3
"""Cross-machine environment smoke test. Run the same script on old server and AutoDL."""

from __future__ import annotations

import os
import platform
import subprocess
import sys


def _run(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as e:
        return f"FAIL: {e}"


def main():
    print("=" * 60)
    print("ConverSeg env smoke test")
    print("=" * 60)

    print(f"python: {sys.version}")
    print(f"executable: {sys.executable}")
    print(f"platform: {platform.platform()}")
    print(f"ldd glibc: {_run('ldd --version | head -1')}")

    # torch
    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")
        print(f"CUDA_VISIBLE_DEVICES: {cvd}")
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            print(f"torch.cuda.device_count: {n}")
            for i in range(n):
                print(f"torch.cuda.get_device_name({i}): {torch.cuda.get_device_name(i)}")
            for i in range(n):
                try:
                    t = torch.zeros(1, device=f"cuda:{i}")
                    torch.cuda.synchronize(f"cuda:{i}")
                    print(f"cuda:{i} alloc ok")
                except Exception as e:
                    print(f"cuda:{i} alloc FAIL: {e}")
        x = torch.randn(2, 3)
        y = x @ x.T
        print(f"torch matmul ok, shape={tuple(y.shape)}")
    except Exception as e:
        print(f"torch: FAIL ({e})")

    # common deps
    for pkg in [
        "numpy",
        "cv2",
        "PIL",
        "transformers",
        "peft",
        "hydra",
        "iopath",
    ]:
        try:
            m = __import__(pkg if pkg != "cv2" else "cv2")
            ver = getattr(m, "__version__", "ok")
            print(f"{pkg}: {ver}")
        except Exception as e:
            print(f"{pkg}: FAIL ({e})")

    # sam2
    try:
        import sam2
        from sam2.build_sam import build_sam2
        print(f"sam2: {sam2.__file__}")
        print("sam2 import: OK")
    except Exception as e:
        print(f"sam2: FAIL ({e})")

    print("=" * 60)
    print("Done. Compare this output between machines.")
    print("=" * 60)


if __name__ == "__main__":
    main()
