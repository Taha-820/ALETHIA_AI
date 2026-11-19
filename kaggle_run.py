#!/usr/bin/env python3
"""Unattended full run on Kaggle (or any Linux GPU box): download -> run.py -> commit + push, chunk by chunk.

    HF_TOKEN=... GH_TOKEN=... python kaggle_run.py [--chunk 5] [--budget-hours 11.25]

* Bootstraps a fresh venv from requirements.txt (same as the graders' pip install), runs prepare.py.
* Lists data/train-000[0-4]* of HuggingFaceFV/finevideo (50 files) and processes them in chunks. The next
  chunk downloads while the current one runs; parquet files are deleted once their chunk is done.
* After every chunk: results/parts/<file>.jsonl.gz per parquet file, results/logs/<chunk>.log (run.py output
  with its timing breakdown), results/progress.json, and a rebuilt results.jsonl.gz -> git commit + push.
* Resumable: files that already have a part in results/parts/ are skipped, so if the session hits its time
  limit just run it again. It also stops by itself before starting a chunk that would not fit in the budget.
* Tokens are only read from the environment; GH_TOKEN is passed to git through a credential helper that
  reads the variable, so it is never written to .git/config, the logs or the repository.
"""
from __future__ import annotations

import argparse
import fnmatch
import gzip
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.abspath(__file__))
DATASET = "HuggingFaceFV/finevideo"
PATTERN = "data/train-000[0-4]*"
T0 = float(os.environ.setdefault("ALETHIA_T0", str(time.time())))  # survives the re-exec into the venv


def log(msg):
    print(f"[kaggle_run {time.strftime('%H:%M:%S')} +{(time.time() - T0) / 3600:5.2f}h] {msg}", flush=True)


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


# ----------------------------------------------------------------------------------------------- git
GIT_AUTH = ["-c", "credential.helper=",
            "-c", "credential.helper=!f() { echo username=x-access-token; echo password=$GH_TOKEN; }; f"]


def git(*args, check=True, capture=False):
    name = os.environ.get("GIT_NAME", "Taha820")
    email = os.environ.get("GIT_EMAIL", "Taha820@users.noreply.github.com")
    cmd = ["git", *GIT_AUTH, "-c", f"user.name={name}", "-c", f"user.email={email}", *args]
    return subprocess.run(cmd, cwd=REPO, check=check, text=True,
                          stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None)


def push(message, extra=()):
    git("add", "results", "results.jsonl.gz", *extra)
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        log("nothing new to commit")
        return
    git("commit", "-q", "-m", message)
    for attempt in range(5):
        r = git("push", "-q", "origin", "HEAD", check=False, capture=True)
        if r.returncode == 0:
            log(f"pushed: {message}")
            return
        log(f"push failed (attempt {attempt + 1}): {r.stdout.strip()[-300:]}; rebasing and retrying")
        git("pull", "-q", "--rebase", "origin", git("rev-parse", "--abbrev-ref", "HEAD", capture=True).stdout.strip(),
            check=False)
        time.sleep(10 * (attempt + 1))
    log("WARNING: could not push; results are committed locally and will go up with the next push")


# ----------------------------------------------------------------------------------------------- stage 1
def bootstrap(args):
    """Runs under the system python: apt packages, fresh venv, pip install -r requirements.txt, prepare.py."""
    if not os.environ.get("HF_TOKEN") or not (args.smoke_test or os.environ.get("GH_TOKEN")):
        sys.exit("set HF_TOKEN (and GH_TOKEN for the full run) in the environment (Kaggle: Add-ons > Secrets)")
    sudo = [] if os.geteuid() == 0 else ["sudo"]
    r = subprocess.run(sudo + ["apt-get", "install", "-y", "-qq", "libgl1", "libglib2.0-0"], capture_output=True)
    if r.returncode != 0:
        subprocess.run(sudo + ["apt-get", "update", "-qq"], capture_output=True)
        subprocess.run(sudo + ["apt-get", "install", "-y", "-qq", "libgl1", "libglib2.0-0"], capture_output=True)
    py = sys.executable
    if not args.no_venv:
        venv = args.venv
        py = os.path.join(venv, "bin", "python")
        if not os.path.exists(os.path.join(venv, ".installed")):
            log(f"creating venv {venv} and installing requirements.txt (a few minutes)")
            try:
                sh([sys.executable, "-m", "venv", venv])
                sh([py, "-m", "pip", "install", "-q", "--upgrade", "pip"])
                sh([py, "-m", "pip", "install", "-q", "-r", os.path.join(REPO, "requirements.txt")])
                open(os.path.join(venv, ".installed"), "w").close()
            except subprocess.CalledProcessError as e:
                log(f"venv setup failed ({e}); installing requirements.txt into the current environment instead")
                py = sys.executable
                sh([py, "-m", "pip", "install", "-q", "-r", os.path.join(REPO, "requirements.txt")])
    sh([py, os.path.join(REPO, "prepare.py")])
    ok = subprocess.run([py, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"]).returncode
    if ok != 0:
        sys.exit("torch cannot see a GPU in this environment (driver too old for CUDA 12.8, or no GPU attached)")
    os.execv(py, [py, os.path.abspath(__file__), "--stage2", *sys.argv[1:]])


# ----------------------------------------------------------------------------------------------- stage 2
def hardware_report(path):
    out = [f"date: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}", f"python: {platform.python_version()}",
           f"platform: {platform.platform()}"]
    for title, cmd in [("nvidia-smi", ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                       "--format=csv"]),
                       ("lscpu", ["lscpu"]), ("memory", ["free", "-g"]), ("disk", ["df", "-h", "/tmp", REPO]),
                       ("disk type (ROTA=1 means spinning disk)", ["lsblk", "-d", "-o", "NAME,ROTA,SIZE,MODEL"]),
                       ("os", ["cat", "/etc/os-release"])]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            out += [f"\n## {title}", r.stdout.strip()]
        except Exception as e:
            out += [f"\n## {title}", f"unavailable: {e}"]
    try:
        import torch

        out += ["\n## torch", f"{torch.__version__} CUDA {torch.version.cuda}"]
    except Exception:
        pass
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def list_files(token, n):
    from huggingface_hub import HfApi

    files = HfApi(token=token).list_repo_files(DATASET, repo_type="dataset")
    files = sorted(f for f in files if fnmatch.fnmatch(f, PATTERN) and f.endswith(".parquet"))
    if len(files) != 50:
        log(f"WARNING: pattern {PATTERN} matched {len(files)} files (expected 50)")
    return files[:n]


class Downloader(threading.Thread):
    """Downloads files in order, staying at most `ahead` files beyond the one being processed."""

    def __init__(self, files, dest, token, ahead):
        super().__init__(daemon=True)
        self.files, self.dest, self.token, self.ahead = files, dest, token, ahead
        self.paths: dict[str, str | None] = {}
        self.cv = threading.Condition()
        self.consumed = 0

    def run(self):
        from huggingface_hub import hf_hub_download

        for i, f in enumerate(self.files):
            with self.cv:
                while i >= self.consumed + self.ahead:
                    self.cv.wait()
            path = None
            for attempt in range(5):
                try:
                    t = time.time()
                    path = hf_hub_download(DATASET, f, repo_type="dataset", local_dir=self.dest, token=self.token)
                    log(f"downloaded {os.path.basename(f)} ({os.path.getsize(path) / 2**20:.0f} MiB, "
                        f"{time.time() - t:.0f}s)")
                    break
                except Exception as e:
                    log(f"download of {f} failed (attempt {attempt + 1}): {e}")
                    time.sleep(30 * (attempt + 1))
            with self.cv:
                self.paths[f] = path
                self.cv.notify_all()

    def get(self, f):
        with self.cv:
            while f not in self.paths:
                self.cv.wait()
            return self.paths[f]

    def done_with(self, n):
        with self.cv:
            self.consumed += n
            self.cv.notify_all()


def run_chunk(paths, chunk_dir, out_path, log_path, extra, timing_path=None):
    shutil.rmtree(chunk_dir, ignore_errors=True)
    os.makedirs(chunk_dir)
    for p in paths:
        os.symlink(os.path.abspath(p), os.path.join(chunk_dir, os.path.basename(p)))
    cmd = [sys.executable, os.path.join(REPO, "run.py"), "--input", chunk_dir, "--output", out_path, *extra]
    if timing_path:
        cmd += ["--timing-json", timing_path]
    with open(log_path, "a") as lf:
        lf.write(f"$ {' '.join(cmd)}\n")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
        return p.wait()


def rebuild_results(parts_dir, dest):
    tmp = dest + ".tmp"
    n = 0
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=9) as out:
        for name in sorted(os.listdir(parts_dir)):
            if name.endswith(".jsonl.gz"):
                with gzip.open(os.path.join(parts_dir, name), "rt", encoding="utf-8") as f:
                    for line in f:
                        out.write(line)
                        n += 1
    os.replace(tmp, dest)
    return n


def validation_reference(path, out, args, logs_dir):
    """Every-frame FP32 reference with the official predictor on the first videos of the first file."""
    if path is None:
        return
    vdir = os.path.join(args.work, "val")
    shutil.rmtree(vdir, ignore_errors=True)
    os.makedirs(vdir)
    os.symlink(os.path.abspath(path), os.path.join(vdir, os.path.basename(path)))
    cmd = [sys.executable, os.path.join(REPO, "tests", "reference.py"), "--input", vdir, "--output", out,
           "--device", args.validate_device, "--max-videos", str(args.validate_videos), "--max-seconds", str(args.validate_seconds)]
    log(f"validation: reference (every frame, FP32, official predictor) on {args.validate_videos} videos x "
        f"{args.validate_seconds}s")
    t = time.time()
    with open(os.path.join(logs_dir, "validation_reference.log"), "w") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    log(f"validation reference finished in {(time.time() - t) / 60:.1f} min (exit {r.returncode})")


def validation_compare(ref, res, dest, args):
    r = subprocess.run([sys.executable, os.path.join(REPO, "compare.py"), ref, res], capture_output=True, text=True)
    text = (f"run.py output vs. the every-frame FP32 reference (tests/reference.py) on the first "
            f"{args.validate_videos} videos x {args.validate_seconds}s of {os.path.basename(res)}\n"
            f"(only reference seconds are scored)\n\n{r.stdout}{r.stderr}")
    with open(dest, "w") as f:
        f.write(text)
    log("validation result:\n" + text)
    if r.returncode != 0:
        log("WARNING: validation FAILED the brief's thresholds - check results/validation.txt before trusting the run")


# ----------------------------------------------------------------------------------------------- reports
def _cmd(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def _hms(sec):
    sec = int(round(sec))
    return f"{sec // 3600} h {sec % 3600 // 60:02d} min" if sec >= 3600 else f"{sec // 60} min {sec % 60:02d} s"


def _load_timings(logs_dir):
    out = []
    for name in sorted(os.listdir(logs_dir)):
        if name.endswith(".timing.json"):
            try:
                with open(os.path.join(logs_dir, name)) as f:
                    out.append((name[: -len(".timing.json")], json.load(f)))
            except Exception:
                pass
    return out


def timing_summary(logs_dir, dest):
    """Per-chunk and total timing breakdown of run.py (from its --timing-json output)."""
    rows = _load_timings(logs_dir)
    if not rows:
        return
    keys = ["wall_s", "frames_decoded", "frames_sampled", "frames_inferred", "gpu_busy_s", "gpu_starved_s",
            "gpu_thread_copy_s", "gpu_thread_infer_s", "worker_decode_s", "worker_convert_s", "worker_letterbox_s",
            "worker_slot_wait_s", "model_load_s"]
    tot = {k: sum(r.get(k) or 0 for _, r in rows) for k in keys}
    pct = lambda a, b: f"{100 * a / b:.1f}%" if b else "-"
    first = rows[0][1]
    nw = first.get("workers", 1)
    lines = [
        "# run.py timing breakdown (measured on this machine; see HARDWARE.md)", "",
        f"Settings: {first.get('sample_fps')} fps sampling, {first.get('precision')}, batch {first.get('batch')}, "
        f"{nw} decode workers, devices {', '.join(first.get('devices', []))} x {first.get('threads_per_device')} "
        f"GPU threads each.", "",
        "| chunk | wall | frames decoded | frames inferred | inferred/s | GPU busy | GPU starved (decode wait) |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, r in rows:
        w = r.get("wall_s") or 0
        lines.append(f"| {label} | {_hms(w)} | {r.get('frames_decoded')} | {r.get('frames_inferred')} | "
                     f"{(r.get('frames_inferred') or 0) / max(w, 1e-9):.1f} | {pct(r.get('gpu_busy_s') or 0, w)} | "
                     f"{pct(r.get('gpu_starved_s') or 0, w)} |")
    w = tot["wall_s"]
    lines += [
        f"| **total** | **{_hms(w)}** | {tot['frames_decoded']} | {tot['frames_inferred']} | "
        f"{tot['frames_inferred'] / max(w, 1e-9):.1f} | {pct(tot['gpu_busy_s'], w)} | {pct(tot['gpu_starved_s'], w)} |",
        "",
        "Totals over all chunks:",
        "",
        f"* GPU busy (at least one GPU thread working): {_hms(tot['gpu_busy_s'])} ({pct(tot['gpu_busy_s'], w)} of wall)",
        f"* GPU starved, i.e. waiting for decoded frames: {_hms(tot['gpu_starved_s'])} ({pct(tot['gpu_starved_s'], w)})",
        f"* Model loading: {_hms(tot['model_load_s'])}",
        f"* GPU threads, summed over threads: host copy {_hms(tot['gpu_thread_copy_s'])}, "
        f"forward + NMS {_hms(tot['gpu_thread_infer_s'])}",
        f"* Decode workers, summed over {nw} processes (divide by {nw} for wall-clock share): "
        f"demux+decode {_hms(tot['worker_decode_s'])}, BGR conversion {_hms(tot['worker_convert_s'])}, "
        f"rotate+letterbox {_hms(tot['worker_letterbox_s'])}, blocked waiting for free slots "
        f"(GPU back-pressure) {_hms(tot['worker_slot_wait_s'])}",
        "",
        "How to read it: when 'GPU starved' is large and 'blocked waiting for free slots' is near zero, decoding is "
        "the bottleneck; when the GPU is busy almost all the time and the workers spend time blocked on slots, "
        "the GPU is the bottleneck.",
    ]
    with open(dest, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_hardware_md(progress, n_files, dest):
    """Facts for the brief's HARDWARE.md, collected on the machine that did the run."""
    gpus = [g.strip() for g in _cmd(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
            .splitlines() if g.strip()]
    driver = _cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).splitlines()
    lscpu = {}
    for line in _cmd(["lscpu"]).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            lscpu[k.strip()] = v.strip()
    try:
        usable = len(os.sched_getaffinity(0))
    except Exception:
        usable = os.cpu_count()
    mem_kb = 0
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                mem_kb = int(line.split()[1])
    except Exception:
        pass
    cg_limit = None
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(path).read().strip()
            if v.isdigit() and int(v) < mem_kb * 1024:
                cg_limit = int(v)
        except Exception:
            pass
    disks = []
    for line in _cmd(["lsblk", "-d", "-n", "-b", "-o", "NAME,ROTA,TYPE,SIZE"]).splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[2] == "disk" and parts[3].isdigit() and int(parts[3]) >= 2**30:
            disks.append((parts[0], parts[1], f"{int(parts[3]) / 2**30:.0f} GiB"))
    if disks and all(r == "0" for _, r, _ in disks):
        disk_type = "SSD (lsblk reports non-rotational: " + ", ".join(f"{n} {s}" for n, _, s in disks) + ")"
    elif disks:
        disk_type = ("not certain: lsblk reports " + ", ".join(f"{n} {s} ROTA={r}" for n, r, s in disks)
                     + " (ROTA=1 means rotational, but virtual cloud disks often report 1 even on SSD storage)")
    else:
        disk_type = "unknown (lsblk shows no block devices inside the container)"
    df = _cmd(["df", "-h", "--output=source,fstype,size", "/tmp"]).splitlines()
    os_name = "unknown"
    try:
        for line in open("/etc/os-release"):
            if line.startswith("PRETTY_NAME="):
                os_name = line.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    try:
        import torch

        torch_s = f"torch {torch.__version__} (CUDA {torch.version.cuda})"
    except Exception:
        torch_s = "torch unavailable"
    where = "Kaggle notebook (free tier, accelerator GPU T4 x2)" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") \
        or os.path.isdir("/kaggle") else platform.node()
    sessions = progress.get("sessions", [])
    sess_wall = sum(x["end_epoch"] - x["start_epoch"] for x in sessions)
    runpy = sum(c["seconds"] for c in progress.get("chunks", []))
    start = min((x["start_epoch"] for x in sessions), default=None)
    end = max((x["end_epoch"] for x in sessions), default=None)
    fmt = lambda e: time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(e))
    lines = [
        "# Hardware",
        "",
        "Collected automatically by `kaggle_run.py` on the machine that processed the 50 files.",
        "",
        f"* **Where**: {where}",
        f"* **GPU**: " + ("; ".join(f"{g}" for g in gpus) if gpus else "unknown")
        + (f" ({len(gpus)} GPUs, NVIDIA driver {driver[0].strip()})" if gpus and driver else ""),
        f"* **CPU**: {lscpu.get('Model name', 'unknown')}, {lscpu.get('CPU(s)', '?')} logical CPUs "
        f"({lscpu.get('Thread(s) per core', '?')} thread(s) per core, {lscpu.get('Core(s) per socket', '?')} core(s) "
        f"per socket, {lscpu.get('Socket(s)', '?')} socket(s)); {usable} usable by the run",
        f"* **RAM**: {mem_kb / 2**20:.1f} GiB" + (f" (container limit {cg_limit / 2**30:.1f} GiB)" if cg_limit else ""),
        f"* **Disk**: {disk_type}" + (f"; scratch space for downloads: {' '.join(df[-1].split())}" if len(df) > 1 else ""),
        f"* **OS**: {os_name}, Linux kernel {platform.release()} (inside the platform's container)",
        f"* **Software**: Python {platform.python_version()}, {torch_s}",
        "",
        "## How long the full run took",
        "",
        f"* **{_hms(runpy)}** in total inside `run.py` for all {n_files} files (sum of the per-chunk run.py "
        f"invocations; per-chunk numbers in `results/progress.json` and `results/timing_summary.md`).",
        f"* {_hms(sess_wall)} of total session wall time, from session start to the last push, including "
        f"environment setup, dataset downloads (overlapped with processing), the validation reference and "
        f"git pushes, over {len(sessions)} session(s)"
        + (f" between {fmt(start)} and {fmt(end)}." if start else "."),
        "",
        "Note: the files were processed in chunks of a few Parquet files per `run.py` invocation so results could "
        "be pushed as the run went; each invocation adds a few seconds of start-up (model load, worker start).",
    ]
    with open(dest, "w") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------------------------- smoke test
SMOKE_VARIANTS = [  # (label, run.py args, the one-line change in run.py that makes it the default)
    ("5 fps, FP16 (current defaults)", [], None),
    ("10 fps, FP16", ["--sample-fps", "10"],
     ("--sample-fps\", type=int, default=5,", "--sample-fps\", type=int, default=10,")),
    ("5 fps, FP32", ["--fp32"],
     ("\"--fp32\", action=\"store_true\",", "\"--fp32\", action=\"store_true\", default=True,")),
    ("every frame, FP16", ["--sample-fps", "0"],
     ("--sample-fps\", type=int, default=5,", "--sample-fps\", type=int, default=0,")),
]


def smoke_verdict(results):
    """results: [(label, metrics, wall)] in SMOKE_VARIANTS order -> (exit code, verdict text)."""
    out = []
    print = lambda *a: out.append(" ".join(str(x) for x in a))  # noqa: A001 - collect instead of printing
    margin = lambda m: m["f1"] >= 0.97 and m["exact"] >= 0.90 and m["within1"] >= 0.97
    default = results[0][1]
    if default["pass"] and margin(default):
        print("VERDICT: PASS with margin. Launch the full run as is.")
        code = 0
    else:
        better = [(lab, m, ch) for (lab, m, _), (_, _, ch) in zip(results[1:], SMOKE_VARIANTS[1:]) if m["pass"]]
        if default["pass"]:
            print("VERDICT: PASS, but with a thin margin (want roughly F1 >= 0.97, exact >= 0.90, within1 >= 0.97 "
                  "because the hidden test set differs). Launching as is is allowed; safer option below.")
            code = 0
        else:
            print("VERDICT: FAIL with the current defaults. Do NOT launch as is.")
            code = 1
        if better:
            lab, m, (old, new) = max(better, key=lambda x: (margin(x[1]), -[v[0] for v in SMOKE_VARIANTS].index(x[0])))
            print(f"Recommended: '{lab}' (F1 {m['f1']:.4f}, exact {m['exact']:.4f}, within1 {m['within1']:.4f}).")
            print("One-line change in run.py (then commit, push and run this smoke test again):")
            print(f"    sed -i 's/{old}/{new}/' run.py")
        elif not default["pass"]:
            print("No tested setting passes, not even every frame. That points at the pipeline, not the sampling "
                  "rate: keep this output and debug before launching.")
    return code, "\n".join(out)


def smoke_test(args):
    """Interactive 10-15 minute check before the long unattended run (no git, nothing pushed)."""
    import compare

    token = os.environ.get("HF_TOKEN")
    work = os.path.join(args.work, "smoke")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    log("GPU check")
    print(_cmd(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv"]), flush=True)
    import torch

    for i in range(torch.cuda.device_count()):
        a = torch.randn(512, 512, device=f"cuda:{i}", dtype=torch.half)
        ok = torch.isfinite(a @ a).all().item()
        print(f"  cuda:{i} {torch.cuda.get_device_name(i)}: FP16 matmul {'ok' if ok else 'NOT OK'}", flush=True)
    print(f"  {torch.__version__}, CUDA runtime {torch.version.cuda}", flush=True)

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    f0 = list_files(token, 1)[0]
    t = time.time()
    p0 = hf_hub_download(DATASET, f0, repo_type="dataset", local_dir=os.path.join(args.work, "dl"), token=token)
    log(f"downloaded {os.path.basename(f0)} in {time.time() - t:.0f}s")
    sub_dir = os.path.join(work, "input")
    os.makedirs(sub_dir)
    table = pq.read_table(p0).slice(0, args.smoke_videos)  # first rows -> keys file:0..N-1 match the real file
    pq.write_table(table, os.path.join(sub_dir, os.path.basename(p0)))
    del table
    ref = os.path.join(work, "reference.jsonl")
    log(f"reference: official YOLO.predict, FP32, every frame, first {args.smoke_seconds}s of {args.smoke_videos} "
        f"videos (the slow part, ~5 min)")
    t = time.time()
    r = subprocess.run([sys.executable, os.path.join(REPO, "tests", "reference.py"), "--input", sub_dir, "--output", ref,
                        "--device", args.validate_device, "--max-seconds", str(args.smoke_seconds)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        sys.exit("reference failed")
    log(f"reference done in {(time.time() - t) / 60:.1f} min")

    results = []
    for i, (label, extra, _) in enumerate(SMOKE_VARIANTS):
        out = os.path.join(work, f"run{i}.jsonl")
        tj = os.path.join(work, f"run{i}.timing.json")
        cmd = [sys.executable, os.path.join(REPO, "run.py"), "--input", sub_dir, "--output", out,
               "--max-seconds", str(args.smoke_seconds), "--timing-json", tj, *extra]
        t = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.time() - t
        if i == 0:  # show the full run.py output once, including the timing breakdown
            print(r.stderr, flush=True)
        if r.returncode != 0:
            print(r.stderr[-4000:], flush=True)
            log(f"run.py FAILED for '{label}' (exit {r.returncode}) - this is the CUDA/FP16 path; do not launch")
            sys.exit(1)
        m = compare.score(ref, out)
        results.append((label, m, wall))
        log(f"{label}: F1 {m['f1']:.4f}, people exact {m['exact']:.4f}, within 1 {m['within1']:.4f} "
            f"-> {'PASS' if m['pass'] else 'FAIL'} ({wall:.0f}s)")

    print("\n" + "=" * 100)
    print(f"SMOKE TEST: {args.smoke_videos} videos x first {args.smoke_seconds}s of {os.path.basename(f0)}, "
          f"scored against the every-frame FP32 reference ({results[0][1]['seconds']} seconds)")
    print(f"{'setting':34s} {'obj F1':>8s} {'exact':>8s} {'within1':>8s}   result   (thresholds 0.95 / 0.85 / 0.95)")
    for label, m, wall in results:
        print(f"{label:34s} {m['f1']:8.4f} {m['exact']:8.4f} {m['within1']:8.4f}   {'PASS' if m['pass'] else 'FAIL'}")
    print("\nDefault settings in detail:\n" + compare.report(results[0][1]))
    code, text = smoke_verdict(results)
    print("\n" + text)
    print("=" * 100, flush=True)
    sys.exit(code)


def stage2(args):
    token = os.environ.get("HF_TOKEN")
    results = os.path.join(REPO, "results")
    parts_dir, logs_dir = os.path.join(results, "parts"), os.path.join(results, "logs")
    os.makedirs(parts_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    hardware_report(os.path.join(results, "kaggle_hardware.txt"))
    prog_path = os.path.join(results, "progress.json")
    progress = json.load(open(prog_path)) if os.path.exists(prog_path) else {"chunks": []}
    session = {"started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(T0)), "start_epoch": round(T0, 1),
               "end_epoch": round(time.time(), 1), "files_processed": 0}
    progress.setdefault("sessions", []).append(session)

    def save_progress():
        session["end_epoch"] = round(time.time(), 1)
        progress["total_seconds_in_run_py"] = round(sum(c["seconds"] for c in progress["chunks"]), 1)
        progress["total_session_wall_seconds"] = round(sum(x["end_epoch"] - x["start_epoch"]
                                                           for x in progress["sessions"]), 1)
        with open(prog_path, "w") as f:
            json.dump(progress, f, indent=1)

    def finish():
        save_progress()
        timing_summary(logs_dir, os.path.join(results, "timing_summary.md"))
        write_hardware_md(progress, len(files), os.path.join(REPO, "HARDWARE.md"))
        n = rebuild_results(parts_dir, os.path.join(REPO, "results.jsonl.gz"))
        push(f"results: all {len(files)} files ({n} lines) + HARDWARE.md", extra=["HARDWARE.md"])

    files = list_files(token, args.files)
    stem = lambda f: os.path.basename(f)[: -len(".parquet")]
    todo = [f for f in files if not os.path.exists(os.path.join(parts_dir, stem(f) + ".jsonl.gz"))]
    log(f"{len(files) - len(todo)} of {len(files)} files already done; {len(todo)} to go")
    if not todo:
        finish()
        return
    chunks = [todo[i : i + args.chunk] for i in range(0, len(todo), args.chunk)]
    dl_dir = os.path.join(args.work, "dl")
    dl = Downloader(todo, dl_dir, token, ahead=2 * args.chunk)
    dl.start()
    extra = args.run_args.split() if args.run_args else []
    durations = [c["seconds"] for c in progress["chunks"]]
    val_ref = os.path.join(results, "validation_reference.jsonl")
    val_file = None
    if args.validate_videos and not os.path.exists(os.path.join(results, "validation.txt")):
        val_file = todo[0]
        validation_reference(dl.get(val_file), val_ref, args, logs_dir)
    for ci, chunk in enumerate(chunks):
        elapsed = time.time() - T0
        est = max(durations) * 1.15 if durations else 0
        if elapsed + est > args.budget_hours * 3600:
            log(f"stopping: {elapsed / 3600:.2f}h used, next chunk needs ~{est / 3600:.2f}h of the "
                f"{args.budget_hours}h budget. Run this again to continue with {sum(map(len, chunks[ci:]))} files.")
            return
        paths = [dl.get(f) for f in chunk]
        ok_paths = [p for p in paths if p]
        names = [stem(f) for f in chunk]
        label = f"{names[0]}..{names[-1]}" if len(names) > 1 else names[0]
        log(f"chunk {ci + 1}/{len(chunks)}: {label}")
        out_path = os.path.join(args.work, "chunk.jsonl")
        log_path = os.path.join(logs_dir, f"{label}.log")
        t = time.time()
        timing_path = os.path.join(logs_dir, f"{label}.timing.json")
        code = run_chunk(ok_paths, os.path.join(args.work, "chunk"), out_path, log_path, extra, timing_path) \
            if ok_paths else 0
        if code != 0:
            log(f"run.py exited with {code}; retrying the chunk once")
            code = run_chunk(ok_paths, os.path.join(args.work, "chunk"), out_path, log_path, extra, timing_path)
        if code != 0:  # run.py handles bad videos itself, so this is a systemic failure: stop, mark nothing done
            shutil.copy(log_path, log_path + ".FAILED")
            git("add", "results/logs", check=False)
            git("commit", "-q", "-m", f"run.py failed on {label}", check=False)
            git("push", "-q", "origin", "HEAD", check=False)
            sys.exit(f"run.py failed twice on {label} (exit {code}); see {log_path}. Nothing marked done.")
        secs = time.time() - t
        by_file: dict[str, list[str]] = {n: [] for n in names}
        if os.path.exists(out_path):
            with open(out_path, encoding="utf-8") as f:
                for line in f:
                    v = json.loads(line)["video"].split(":")[0]
                    by_file.setdefault(v[: -len(".parquet")] if v.endswith(".parquet") else v, []).append(line)
        for f, p in zip(chunk, paths):
            if p is None:
                by_file[stem(f)].append(json.dumps({"video": os.path.basename(f), "error": "download failed"}) + "\n")
        for n, lines in by_file.items():
            with gzip.open(os.path.join(parts_dir, n + ".jsonl.gz"), "wt", encoding="utf-8") as g:
                g.writelines(lines)
        for p in ok_paths:
            try:
                os.remove(p)
            except OSError:
                pass
        dl.done_with(len(chunk))
        durations.append(secs)
        progress["chunks"].append({"files": names, "seconds": round(secs, 1), "exit_code": code,
                                   "lines": sum(len(v) for v in by_file.values()),
                                   "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
        session["files_processed"] += len(chunk)
        save_progress()
        timing_summary(logs_dir, os.path.join(results, "timing_summary.md"))
        if val_file in chunk and os.path.exists(val_ref):
            validation_compare(val_ref, os.path.join(parts_dir, stem(val_file) + ".jsonl.gz"),
                               os.path.join(results, "validation.txt"), args)
        n = rebuild_results(parts_dir, os.path.join(REPO, "results.jsonl.gz"))
        done = len([x for x in os.listdir(parts_dir) if x.endswith(".jsonl.gz")])
        log(f"chunk took {secs / 60:.1f} min (exit {code}); {done}/{len(files)} files done")
        if done >= len(files):
            finish()
        else:
            push(f"results: {done}/{len(files)} files ({n} lines)")
    log(f"all chunks finished in {(time.time() - T0) / 3600:.2f}h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=5, help="parquet files per run.py invocation / push")
    ap.add_argument("--files", type=int, default=50)
    ap.add_argument("--budget-hours", type=float, default=11.25, help="do not start a chunk past this")
    ap.add_argument("--work", default="/tmp/alethia_work", help="download + scratch dir (not /kaggle/working)")
    ap.add_argument("--venv", default="/tmp/alethia_venv")
    ap.add_argument("--no-venv", action="store_true", help="use the current python environment as is")
    ap.add_argument("--run-args", default="", help="extra arguments for run.py, e.g. '--batch 32'")
    ap.add_argument("--validate-videos", type=int, default=6, help="0 disables the reference check")
    ap.add_argument("--validate-seconds", type=int, default=60)
    ap.add_argument("--validate-device", default="0")
    ap.add_argument("--smoke-test", action="store_true",
                    help="10-15 min check: CUDA/FP16 path + accuracy vs. the reference on a few videos; no git")
    ap.add_argument("--smoke-videos", type=int, default=6)
    ap.add_argument("--smoke-seconds", type=int, default=60)
    ap.add_argument("--stage2", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    if args.stage2:
        smoke_test(args) if args.smoke_test else stage2(args)
    else:
        bootstrap(args)


if __name__ == "__main__":
    main()
