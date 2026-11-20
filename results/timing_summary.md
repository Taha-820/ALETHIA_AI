# run.py timing breakdown (measured on this machine; see HARDWARE.md)

Settings: 10 fps sampling, fp16, batch 32, 3 decode workers, devices cuda:0, cuda:1 x 2 GPU threads each.

| chunk | wall | frames decoded | frames inferred | inferred/s | GPU busy | GPU starved (decode wait) |
|---|---|---|---|---|---|---|
| train-00000-of-01357..train-00004-of-01357 | 47 min 50 s | 1367944 | 486703 | 169.6 | 99.7% | 0.0% |
| train-00005-of-01357..train-00009-of-01357 | 41 min 17 s | 1214802 | 433445 | 175.0 | 99.5% | 0.0% |
| train-00010-of-01357..train-00014-of-01357 | 47 min 54 s | 1403768 | 495595 | 172.4 | 99.5% | 0.0% |
| train-00015-of-01357..train-00019-of-01357 | 47 min 16 s | 1387088 | 492185 | 173.5 | 99.6% | 0.0% |
| train-00020-of-01357..train-00024-of-01357 | 43 min 04 s | 1248344 | 446653 | 172.9 | 99.5% | 0.0% |
| **total** | **3 h 47 min** | 6621946 | 2354581 | 172.6 | 99.6% | 0.0% |

Totals over all chunks:

* GPU busy (at least one GPU thread working): 3 h 46 min (99.6% of wall)
* GPU starved, i.e. waiting for decoded frames: 0 min 03 s (0.0%)
* Model loading: 0 min 59 s
* GPU threads, summed over threads: host copy 13 min 16 s, forward + NMS 14 h 51 min
* Decode workers, summed over 3 processes (divide by 3 for wall-clock share): demux+decode 2 h 22 min, BGR conversion 2 h 43 min, rotate+letterbox 8 min 23 s, blocked waiting for free slots (GPU back-pressure) 5 h 33 min

How to read it: when 'GPU starved' is large and 'blocked waiting for free slots' is near zero, decoding is the bottleneck; when the GPU is busy almost all the time and the workers spend time blocked on slots, the GPU is the bottleneck.
