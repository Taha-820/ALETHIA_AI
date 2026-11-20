# run.py timing breakdown (measured on this machine; see HARDWARE.md)

Settings: 10 fps sampling, fp16, batch 32, 3 decode workers, devices cuda:0, cuda:1 x 2 GPU threads each.

| chunk | wall | frames decoded | frames inferred | inferred/s | GPU busy | GPU starved (decode wait) |
|---|---|---|---|---|---|---|
| train-00000-of-01357..train-00004-of-01357 | 47 min 50 s | 1367944 | 486703 | 169.6 | 99.7% | 0.0% |
| train-00005-of-01357..train-00009-of-01357 | 41 min 17 s | 1214802 | 433445 | 175.0 | 99.5% | 0.0% |
| train-00010-of-01357..train-00014-of-01357 | 47 min 54 s | 1403768 | 495595 | 172.4 | 99.5% | 0.0% |
| train-00015-of-01357..train-00019-of-01357 | 47 min 16 s | 1387088 | 492185 | 173.5 | 99.6% | 0.0% |
| train-00020-of-01357..train-00024-of-01357 | 43 min 04 s | 1248344 | 446653 | 172.9 | 99.5% | 0.0% |
| train-00025-of-01357..train-00029-of-01357 | 43 min 50 s | 1333625 | 469172 | 178.4 | 99.6% | 0.0% |
| train-00030-of-01357..train-00034-of-01357 | 42 min 17 s | 1271872 | 447360 | 176.3 | 99.6% | 0.0% |
| train-00035-of-01357..train-00039-of-01357 | 41 min 09 s | 1192388 | 422359 | 171.1 | 99.6% | 0.0% |
| train-00040-of-01357..train-00044-of-01357 | 47 min 14 s | 1396450 | 492677 | 173.8 | 99.6% | 0.0% |
| **total** | **6 h 41 min** | 11816281 | 4186149 | 173.6 | 99.6% | 0.0% |

Totals over all chunks:

* GPU busy (at least one GPU thread working): 6 h 40 min (99.6% of wall)
* GPU starved, i.e. waiting for decoded frames: 0 min 05 s (0.0%)
* Model loading: 1 min 42 s
* GPU threads, summed over threads: host copy 23 min 37 s, forward + NMS 26 h 15 min
* Decode workers, summed over 3 processes (divide by 3 for wall-clock share): demux+decode 4 h 13 min, BGR conversion 4 h 52 min, rotate+letterbox 15 min 34 s, blocked waiting for free slots (GPU back-pressure) 9 h 42 min

How to read it: when 'GPU starved' is large and 'blocked waiting for free slots' is near zero, decoding is the bottleneck; when the GPU is busy almost all the time and the workers spend time blocked on slots, the GPU is the bottleneck.
