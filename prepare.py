#!/usr/bin/env python3
"""One-time preparation (needs internet): download the official YOLO11l weights next to run.py.

    python prepare.py
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "yolo11l.pt")
SHA256 = "9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f"  # ultralytics/assets v8.4.0 yolo11l.pt


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if not (os.path.isfile(DEST) and sha256(DEST) == SHA256):
        from ultralytics.utils.downloads import attempt_download_asset

        cwd = os.getcwd()
        os.chdir(HERE)  # attempt_download_asset saves into the working directory
        try:
            attempt_download_asset("yolo11l.pt")
        finally:
            os.chdir(cwd)
    digest = sha256(DEST)
    if digest != SHA256:
        sys.exit(f"yolo11l.pt checksum mismatch: {digest}")
    print(f"yolo11l.pt ready: {DEST}")

    import torch

    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), cuda available: {torch.cuda.is_available()}")
    for i in range(torch.cuda.device_count()):
        print(f"  cuda:{i} {torch.cuda.get_device_name(i)}")


if __name__ == "__main__":
    main()
