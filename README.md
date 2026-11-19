# ALETHIA: per-second object indexing of video with YOLO11l

`run.py` reads FineVideo Parquet files and, for each second of each video, records which of five
objects (person, car, dog, laptop, cell phone) are present and how many people are visible.

## Install (Ubuntu 22.04, Python 3.11, NVIDIA driver with CUDA 12 support)

System packages (OpenCV needs them):

```bash
sudo apt-get install -y libgl1 libglib2.0-0
```

Python packages:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `torch==2.10.0`, the newest PyPI build for CUDA 12 (12.8). Newer PyPI
builds use CUDA 13, which needs a 580-series driver or newer.

## Preparation command (run once, needs internet, not timed)

```bash
python prepare.py
```

This downloads the official `yolo11l.pt` next to `run.py`, checks its SHA-256 and prints the GPUs torch can see.

## Run

```bash
python run.py --input <folder of .parquet files> --output results.jsonl
```

The run itself needs no internet. A timing breakdown is printed to stderr at the end.

Optional flags (the defaults are the intended settings):

| flag | default | meaning |
|---|---|---|
| `--workers` | CPUs − 1 | decode processes |
| `--batch` | 32 | max frames per GPU batch |
| `--sample-fps` | 5 | frames sampled per second of video (`0` = every frame) |
| `--device` | `auto` | all visible GPUs (CPU if none); or `0`, `0,1`, `cpu` |
| `--threads-per-device` | 2 | GPU feeder threads (one model copy each) per GPU |
| `--fp32` | off | disable FP16 |

## How it works

* **Reader process**: streams the `mp4` column out of each Parquet file one row group at a time
  (`pyarrow`), one video's bytes per message. An unreadable file gives one error line with the file
  name; an unreadable row group gives an error line for each of its rows; null or empty `mp4` gives an
  error line.
* **Decode worker processes** (PyAV, one video at a time each): decode every frame, take seconds from
  `pts` with exact integer arithmetic (`(pts − pts_first) · tb_num // tb_den`), and sample the first
  frame in each 1/5-second slot. That gives up to 5 evenly spaced frames per second, and at least 1 in
  every second that has frames. Only sampled frames are converted to BGR. The display-matrix rotation
  or flip is applied the same way as FFmpeg's autorotate. The frame is letterboxed exactly like the
  Ultralytics predictor (`LetterBox(640, auto=True, stride=32)`, bilinear, pad 114) and written into
  a shared-memory slot owned by that worker.
* **Main process**: groups frames by letterboxed shape and sends full batches (or partial ones whenever
  a GPU would otherwise sit idle) to GPU threads. The GPU threads run the same steps as Ultralytics'
  `BasePredictor.preprocess` (BHWC→BCHW, BGR→RGB, FP16, /255), then the Ultralytics `AutoBackend`
  model (fused, FP16) and `non_max_suppression` (conf 0.4, IoU 0.7, class-aware, max_det 300). They
  then count the 5 classes in each frame.
* **Per-second rule**: an object is present if it is found in at least half of the sampled frames of
  that second. `people` is the median count, and an even count gives `(a + b) // 2`. A line is written
  for every second from 0 to the second of the last decoded frame. Seconds without frames are empty.
* **Robustness**: a video that fails part-way keeps its seconds and gets one error line, and so does
  a truncated file. If a decode worker crashes, its video gets an error line and the worker is
  restarted. If a worker stays silent for 300 s, it is killed and restarted. A crashed reader is
  restarted at the next file.

## Checking correctness

`tests/reference.py` is a slow, independent reference. It runs the official `YOLO.predict` on every
frame, as the brief's section 5 describes. `compare.py` scores results against it with the three
pass thresholds:

```bash
python tests/reference.py --input <dir> --output ref.jsonl --device 0 --max-videos 30
python run.py --input <dir> --output res.jsonl
python compare.py ref.jsonl res.jsonl
```

`tests/make_fixtures.py` builds small Parquet files with good and damaged videos for robustness tests.

## Full 50-file run on Kaggle (unattended)

`kaggle_run.py` does the whole run. It sets up a venv from `requirements.txt`, runs `prepare.py`,
downloads the shards a few at a time, runs `run.py` on chunks of 5 files, and then commits and pushes
`results/parts/*.jsonl.gz`, the run logs and a rebuilt `results.jsonl.gz` after every chunk. If the
session ends, run it again: finished files are skipped.

Before the main run it computes the official every-frame FP32 reference on the first 6 videos (60 s
each) of the first shard. After the first chunk it scores `run.py`'s output against that reference and
writes the result to `results/validation.txt`. When all 50 files are done, it also writes
`HARDWARE.md` (GPU, CPU, cores, RAM, disk, OS, total time) and `results/timing_summary.md` (the decode
vs. GPU breakdown summed over all chunks), and pushes them. If `run.py` itself fails twice on a chunk,
the runner stops without marking those files as done.

Notebook settings: GPU T4 x2, Internet on, and the secrets `HF_TOKEN` and `GH_TOKEN` attached.

**Cell 1: setup**

```python
import os, subprocess, sys
from kaggle_secrets import UserSecretsClient
sec = UserSecretsClient()
os.environ["HF_TOKEN"] = sec.get_secret("HF_TOKEN")
os.environ["GH_TOKEN"] = sec.get_secret("GH_TOKEN")

def stream(cmd):  # show a subprocess's output live in the cell / the saved log
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait():
        raise SystemExit(f"exit code {p.returncode}")

helper = "!f() { echo username=x-access-token; echo password=$GH_TOKEN; }; f"
if not os.path.isdir("/kaggle/working/ALETHIA"):
    stream(["git", "-c", f"credential.helper={helper}", "clone",
            "https://github.com/Taha820/ALETHIA.git", "/kaggle/working/ALETHIA"])
else:
    stream(["git", "-C", "/kaggle/working/ALETHIA", "-c", f"credential.helper={helper}", "pull", "-q"])
```

**Cell 2: smoke test (about 10–15 min; run it interactively first)**

```python
stream([sys.executable, "/kaggle/working/ALETHIA/kaggle_run.py", "--smoke-test"])
```

It checks the GPUs and FP16 and builds the every-frame FP32 reference on the first 6 videos × 60 s of
shard 0. It then runs `run.py` four ways (defaults, 10 fps, FP32, every frame), scores each against the
reference and prints a verdict. If the defaults fail or pass only narrowly, it prints the exact
one-line `sed` command that changes `run.py`'s default to the cheapest setting that passes. Nothing is
pushed. It exits with an error on FAIL, so if this cell stays in the notebook it also blocks a "Save &
Run All" from launching a bad run.

**Cell 3: full run (then "Save & Run All")**

```python
stream([sys.executable, "/kaggle/working/ALETHIA/kaggle_run.py"])
```
