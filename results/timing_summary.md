# run.py timing breakdown (measured on this machine; see HARDWARE.md)

Settings: 10 fps sampling, fp16, batch 32, 3 decode workers, devices cuda:0, cuda:1 x 2 GPU threads each.

| chunk | wall | frames decoded | frames inferred | inferred/s | GPU busy | GPU starved (decode wait) |
|---|---|---|---|---|---|---|
| train-00000-of-01357..train-00004-of-01357 | 47 min 50 s | 1367944 | 486703 | 169.6 | 99.7% | 0.0% |
| train-00005-of-01357..train-00009-of-01357 | 41 min 17 s | 1214802 | 433445 | 175.0 | 99.5% | 0.0% |
| **total** | **1 h 29 min** | 2582746 | 920148 | 172.1 | 99.6% | 0.0% |

Totals over all chunks:

* GPU busy (at least one GPU thread working): 1 h 28 min (99.6% of wall)
* GPU starved, i.e. waiting for decoded frames: 0 min 01 s (0.0%)
* Model loading: 0 min 22 s
* GPU threads, summed over threads: host copy 5 min 13 s, forward + NMS 5 h 49 min
* Decode workers, summed over 3 processes (divide by 3 for wall-clock share): demux+decode 57 min 01 s, BGR conversion 1 h 03 min, rotate+letterbox 3 min 15 s, blocked waiting for free slots (GPU back-pressure) 2 h 10 min

How to read it: when 'GPU starved' is large and 'blocked waiting for free slots' is near zero, decoding is the bottleneck; when the GPU is busy almost all the time and the workers spend time blocked on slots, the GPU is the bottleneck.
