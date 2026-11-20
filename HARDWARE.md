# Hardware

Collected automatically by `kaggle_run.py` on the machine that processed the 50 files.

* **Where**: Kaggle notebook (free tier, accelerator GPU T4 x2)
* **GPU**: Tesla T4, 15360 MiB; Tesla T4, 15360 MiB (2 GPUs, NVIDIA driver 580.178.04)
* **CPU**: Intel(R) Xeon(R) CPU @ 2.00GHz, 4 logical CPUs (2 thread(s) per core, 2 core(s) per socket, 1 socket(s)); 4 usable by the run
* **RAM**: 31.3 GiB (container limit 30.0 GiB)
* **Disk**: SSD (lsblk reports non-rotational: sda 8192 GiB, sdb 128 GiB); scratch space for downloads: overlay overlay 8.0T
* **OS**: Ubuntu 24.04.4 LTS, Linux kernel 6.18.55+ (inside the platform's container)
* **Software**: Python 3.13.16, torch 2.10.0+cu128 (CUDA 12.8)

## How long the full run took

* **7 h 23 min** in total inside `run.py` for all 50 files (sum of the per-chunk run.py invocations; per-chunk numbers in `results/progress.json` and `results/timing_summary.md`).
* 7 h 27 min of total session wall time, from session start to the last push, including environment setup, dataset downloads (overlapped with processing), the validation reference and git pushes, over 1 session(s) between 2026-11-19 22:41 UTC and 2026-11-20 06:08 UTC.

Note: the files were processed in chunks of a few Parquet files per `run.py` invocation so results could be pushed as the run went; each invocation adds a few seconds of start-up (model load, worker start).
