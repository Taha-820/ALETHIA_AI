#!/usr/bin/env python3
"""Slow, simple reference: the official Ultralytics predictor on every frame (as in section 5 of the brief).

    python tests/reference.py --input <parquet dir> --output reference.jsonl [--max-videos 20] [--max-seconds 60] [--device 0] [--half]

Frames are decoded with PyAV and rotated the way FFmpeg autorotate does (bit-identical to cv2.VideoCapture
and the ffmpeg CLI for rotation-only display matrices), then passed one by one to YOLO.predict(conf=0.4).
Use it to check run.py: python compare.py reference.jsonl results.jsonl
"""
import argparse, io, json, os, sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run  # rotation helpers and the per-second rule only

import av, cv2, pyarrow.parquet as pq
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--weights", default=os.path.join(os.path.dirname(run.__file__), "yolo11l.pt"))
ap.add_argument("--device", default=None)
ap.add_argument("--half", action="store_true")
ap.add_argument("--sample-fps", type=int, default=0, help="0 = every frame (the graders' reference)")
ap.add_argument("--max-videos", type=int, default=0)
ap.add_argument("--max-seconds", type=int, default=0, help="only the first N seconds of each video")
a = ap.parse_args()

model = YOLO(a.weights)
done = 0
with open(a.output, "w") as out:
    for path in run.list_parquet(a.input):
        name = os.path.basename(path)
        try:
            pf = pq.ParquetFile(path)
            col = pf.read(columns=["mp4"]).column(0)
        except Exception as e:
            print(f"skip {name}: {e}", file=sys.stderr)
            continue
        for row in range(len(col)):
            if a.max_videos and done >= a.max_videos:
                break
            data = col[row].as_py()
            key = f"{name}:{row}"
            if not data:
                continue
            per_sec = {}
            try:
                c = av.open(io.BytesIO(data))
                s = c.streams.video[0]
                pts0, seen = None, set()
                for f in c.decode(s):
                    pts0 = f.pts if pts0 is None else pts0
                    t = Fraction(f.pts - pts0) * s.time_base
                    sec = int(t)  # floor for t >= 0
                    if a.max_seconds and sec >= a.max_seconds:
                        break
                    if a.sample_fps:
                        g = int(t * a.sample_fps)
                        if g in seen:
                            per_sec.setdefault(sec, [])
                            continue
                        seen.add(g)
                    img = run.apply_rotation(f.to_ndarray(format="bgr24"),
                                             run.rotation_op(run.frame_display_matrix(f)), cv2)
                    kw = {"quantize": 16} if a.half else {}
                    r = model.predict(img, conf=0.4, iou=0.7, imgsz=640, device=a.device, verbose=False, **kw)[0]
                    cls = r.boxes.cls.int().tolist()
                    per_sec.setdefault(sec, []).append(tuple(cls.count(k) for k in run.CLASS_IDS))
            except Exception as e:
                print(f"{key}: {e}", file=sys.stderr)
            for sec in range(max(per_sec) + 1 if per_sec else 0):
                objects, people = run.aggregate_second(per_sec.get(sec))
                out.write(json.dumps({"video": key, "second": sec, "objects": objects, "people": people}) + "\n")
            done += 1
            print(f"{key}: {len(per_sec)} s", file=sys.stderr)
