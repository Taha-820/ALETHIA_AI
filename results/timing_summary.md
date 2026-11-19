# run.py timing breakdown (measured on this machine; see HARDWARE.md)

Settings: 10 fps sampling, fp16, batch 32, 3 decode workers, devices cuda:0, cuda:1 x 2 GPU threads each.

| chunk | wall | frames decoded | frames inferred | inferred/s | GPU busy | GPU starved (decode wait) |
|---|---|---|---|---|---|---|
| train-00000-of-01357..train-00004-of-01357 | 47 min 50 s | 1367944 | 486703 | 169.6 | 99.7% | 0.0% |
| **total** | **47 min 50 s** | 1367944 | 486703 | 169.6 | 99.7% | 0.0% |

Totals over all chunks:

* GPU busy (at least one GPU thread working): 47 min 40 s (99.7% of wall)
* GPU starved, i.e. waiting for decoded frames: 0 min 00 s (0.0%)
* Model loading: 0 min 10 s
* GPU threads, summed over threads: host copy 2 min 44 s, forward + NMS 3 h 07 min
* Decode workers, summed over 3 processes (divide by 3 for wall-clock share): demux+decode 29 min 55 s, BGR conversion 32 min 41 s, rotate+letterbox 1 min 44 s, blocked waiting for free slots (GPU back-pressure) 1 h 12 min

How to read it: when 'GPU starved' is large and 'blocked waiting for free slots' is near zero, decoding is the bottleneck; when the GPU is busy almost all the time and the workers spend time blocked on slots, the GPU is the bottleneck.
