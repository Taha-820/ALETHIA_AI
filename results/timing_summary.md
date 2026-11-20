# run.py timing breakdown (measured on this machine; see HARDWARE.md)

Settings: 10 fps sampling, fp16, batch 32, 3 decode workers, devices cuda:0, cuda:1 x 2 GPU threads each.

| chunk | wall | frames decoded | frames inferred | inferred/s | GPU busy | GPU starved (decode wait) |
|---|---|---|---|---|---|---|
| train-00000-of-01357..train-00004-of-01357 | 47 min 50 s | 1367944 | 486703 | 169.6 | 99.7% | 0.0% |
| train-00005-of-01357..train-00009-of-01357 | 41 min 17 s | 1214802 | 433445 | 175.0 | 99.5% | 0.0% |
| train-00010-of-01357..train-00014-of-01357 | 47 min 54 s | 1403768 | 495595 | 172.4 | 99.5% | 0.0% |
| **total** | **2 h 17 min** | 3986514 | 1415743 | 172.2 | 99.6% | 0.0% |

Totals over all chunks:

* GPU busy (at least one GPU thread working): 2 h 16 min (99.6% of wall)
* GPU starved, i.e. waiting for decoded frames: 0 min 02 s (0.0%)
* Model loading: 0 min 36 s
* GPU threads, summed over threads: host copy 8 min 01 s, forward + NMS 8 h 56 min
* Decode workers, summed over 3 processes (divide by 3 for wall-clock share): demux+decode 1 h 26 min, BGR conversion 1 h 37 min, rotate+letterbox 5 min 18 s, blocked waiting for free slots (GPU back-pressure) 3 h 21 min

How to read it: when 'GPU starved' is large and 'blocked waiting for free slots' is near zero, decoding is the bottleneck; when the GPU is busy almost all the time and the workers spend time blocked on slots, the GPU is the bottleneck.
