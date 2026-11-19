#!/usr/bin/env python3
"""ALETHIA - per-second object indexing of FineVideo Parquet shards with YOLO11l.

    python run.py --input <folder of .parquet files> --output results.jsonl

Processes and data flow
-----------------------
    reader process  --(mp4 bytes)-->  main process  --(one video)-->  decode worker k
                                                                         | PyAV decode, sample ~5 fps,
                                                                         | BGR + rotate + letterbox,
                                                                         | write frame into a shared-memory slot
    GPU thread(s) <--(batch of same-shape slots)--  main  <--(slot id)--+
         | copy slots to pinned memory, release slots back to the worker,
         | FP16 forward + Ultralytics NMS, count the 5 classes per frame
         +--(per-frame counts)--> main --> per-second aggregation --> results.jsonl

The main thread is an event loop: it brokers videos to idle workers, groups decoded frames by
letterboxed shape into batches, hands batches to the GPU threads, and writes the output lines of a
video once all its frames are counted. Each decode worker owns a fixed range of shared-memory slots,
so no locks are shared between processes and a crashed worker can be restarted safely.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import queue
import signal
import sys
import tempfile
import threading
import time
import traceback
from collections import deque
from fractions import Fraction
from functools import lru_cache

import multiprocessing as mp
from multiprocessing.connection import wait as mp_wait

import numpy as np

# --------------------------------------------------------------------------------------------- constants
CLASS_IDS = (0, 2, 16, 63, 67)  # COCO ids, in the order of section 1 of the brief
CLASS_NAMES = ("person", "car", "dog", "laptop", "cell phone")
IMGSZ, STRIDE, PAD_VALUE = 640, 32, 114
SLOT_BYTES = IMGSZ * IMGSZ * 3  # a letterboxed frame is at most 640x640x3 (auto=True only shrinks padding)
CONF, IOU, MAX_DET = 0.4, 0.7, 300
MAX_PACKET_ERRORS = 64  # give up on a video after this many packets that fail to decode
MAX_VIDEO_SECONDS = 48 * 3600  # frames with absurd timestamps beyond this are dropped (corrupt pts guard)

perf = time.perf_counter


def log(msg: str) -> None:
    print(f"[run.py {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def short_err(e: BaseException) -> str:
    s = str(e).strip().replace("\n", " ")
    return f"{type(e).__name__}: {s[:160]}" if s else type(e).__name__


# --------------------------------------------------------------------------------------------- letterbox
@lru_cache(maxsize=256)
def letterbox_params(h: int, w: int):
    """Exactly the arithmetic of ultralytics.data.augment.LetterBox(640, auto=True, stride=32, center=True).

    Returns (new_unpad_wh, top, bottom, left, right, out_h, out_w).
    """
    r = min(IMGSZ / h, IMGSZ / w)  # scaleup=True (predictor default)
    new_unpad = (max(round(w * r), 1), max(round(h * r), 1))
    dw, dh = IMGSZ - new_unpad[0], IMGSZ - new_unpad[1]
    dw, dh = np.mod(dw, STRIDE), np.mod(dh, STRIDE)  # auto=True: minimum rectangle
    dw /= 2
    dh /= 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    top, bottom, left, right = int(top), int(bottom), int(left), int(right)
    return new_unpad, top, bottom, left, right, new_unpad[1] + top + bottom, new_unpad[0] + left + right


# --------------------------------------------------------------------------------------------- rotation
@lru_cache(maxsize=64)
def rotation_op(dm_bytes: bytes | None):
    """The transform FFmpeg's autorotate inserts for a display matrix (fftools/ffmpeg_filter.c).

    theta = get_rotation(matrix) = -round(av_display_rotation_get(matrix)), normalised to [0, 360):
      ~90  -> transpose=clock       (or cclock_flip if matrix[3] > 0)
      ~180 -> hflip if matrix[0] < 0, vflip if matrix[4] < 0
      ~270 -> transpose=cclock      (or clock_flip if matrix[3] < 0)
      ~0   -> vflip if matrix[4] < 0
      other -> rotate by theta degrees clockwise, same frame size, black fill
    """
    if not dm_bytes or len(dm_bytes) < 36:
        return None
    m = np.frombuffer(dm_bytes[:36], dtype="=i4").astype(np.float64)
    f = m / 65536.0  # 16.16 fixed point (only the 2x2 rotation/scale part is used)
    s0, s1 = math.hypot(f[0], f[3]), math.hypot(f[1], f[4])
    if s0 == 0.0 or s1 == 0.0:
        return None
    ccw = -math.atan2(f[1] / s1, f[0] / s0) * 180.0 / math.pi  # av_display_rotation_get
    theta = -float(round(ccw))
    theta -= 360.0 * math.floor(theta / 360.0 + 0.9 / 360.0)
    if abs(theta - 90) < 1.0:
        return "cclock_flip" if m[3] > 0 else "clock"
    if abs(theta - 180) < 1.0:
        ops = (("hflip",) if m[0] < 0 else ()) + (("vflip",) if m[4] < 0 else ())
        return ops or None
    if abs(theta - 270) < 1.0:
        return "clock_flip" if m[3] < 0 else "cclock"
    if abs(theta) > 1.0:
        return ("rotate", theta)
    return ("vflip",) if m[4] < 0 else None


def apply_rotation(img, op, cv2):
    if op is None:
        return img
    if op == "clock":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if op == "cclock":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if op == "cclock_flip":  # transpose dir=0: rotate 90 ccw + vertical flip == plain transpose
        return np.ascontiguousarray(img.transpose(1, 0, 2))
    if op == "clock_flip":  # transpose dir=3: rotate 90 cw + vertical flip == anti-transpose
        return np.ascontiguousarray(img[::-1, ::-1].transpose(1, 0, 2))
    if isinstance(op, tuple) and op and op[0] == "rotate":
        h, w = img.shape[:2]
        mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -op[1], 1.0)  # cv2: positive = counter-clockwise
        return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    for o in op:  # tuple of flips
        img = img[:, ::-1] if o == "hflip" else img[::-1]
    return np.ascontiguousarray(img)


def frame_display_matrix(frame) -> bytes | None:
    try:
        for sd in frame.side_data:
            t = sd.type
            if getattr(t, "name", str(t)).endswith("DISPLAYMATRIX"):
                return bytes(sd)
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------------------------- reader process
def reader_main(files: list[tuple[int, str]], conn) -> None:
    """Stream the mp4 column of each Parquet file, one row at a time, to the main process.

    Messages: ('file', idx, name, nrows) ('video', key, bytes|None) ('video_error', key, msg)
              ('file_error', idx, name, msg) ('file_end', idx) ('eof',)
    The pipe blocks when main is not reading, which gives natural back-pressure.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import pyarrow.parquet as pq

    crash_key = os.environ.get("ALETHIA_TEST_READER_CRASH_KEY")  # test hook, see tests/
    for idx, path in files:
        name = os.path.basename(path)
        try:
            pf = pq.ParquetFile(path)
            nrows = pf.metadata.num_rows
            names = pf.schema_arrow.names
        except Exception as e:
            conn.send(("file_error", idx, name, f"could not read parquet file ({short_err(e)})"))
            continue
        conn.send(("file", idx, name, nrows))
        if "mp4" not in names:
            for r in range(nrows):
                conn.send(("video_error", f"{name}:{r}", "no mp4 column in parquet file"))
            conn.send(("file_end", idx))
            continue
        row = 0
        for rg in range(pf.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            try:
                col = pf.read_row_group(rg, columns=["mp4"]).column(0)
            except Exception as e:
                msg = f"could not read parquet row group ({short_err(e)})"
                for i in range(n):
                    conn.send(("video_error", f"{name}:{row + i}", msg))
                row += n
                continue
            for chunk in col.chunks:
                for i in range(len(chunk)):
                    try:
                        v = chunk[i].as_py()
                        if isinstance(v, dict):  # tolerate a struct column like {"bytes": ...}
                            v = v.get("bytes")
                        if isinstance(v, (bytearray, memoryview)):
                            v = bytes(v)
                        if not isinstance(v, bytes) or not v:
                            v = None
                    except Exception:
                        v = None
                    if crash_key == f"{name}:{row}":
                        os.kill(os.getpid(), signal.SIGSEGV)
                    conn.send(("video", f"{name}:{row}", v))
                    row += 1
            del col
        conn.send(("file_end", idx))
    conn.send(("eof",))


# --------------------------------------------------------------------------------------------- decode worker
class _Stop(Exception):
    pass


def worker_main(wid, conn, shm_path, shm_bytes, free_slots, sample_fps, decode_threads, max_seconds=0) -> None:
    """Decode one video at a time. Messages to main: ('frame', key, sec, slot, h, w) ('done', key, max_sec, err, stats)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import av
    import cv2

    cv2.setNumThreads(1)
    try:
        av.logging.set_level(av.logging.PANIC)  # keep FFmpeg quiet about damaged files
    except Exception:
        pass
    buf = np.memmap(shm_path, dtype=np.uint8, mode="r+", shape=(shm_bytes,))
    free = deque(free_slots)
    crash_key = os.environ.get("ALETHIA_TEST_CRASH_KEY")  # test hooks, see tests/
    hang_key = os.environ.get("ALETHIA_TEST_HANG_KEY")
    try:
        while True:
            msg = conn.recv()
            kind = msg[0]
            if kind == "release":
                free.extend(msg[1])
            elif kind == "task":
                decode_video(msg[1], msg[2], conn, buf, free, sample_fps, decode_threads, av, cv2, crash_key, hang_key,
                             max_seconds)
            elif kind == "stop":
                return
    except (_Stop, EOFError, OSError, KeyboardInterrupt):
        return


def decode_video(key, data, conn, buf, free, sample_fps, decode_threads, av, cv2, crash_key=None, hang_key=None,
                 max_seconds=0):
    st = {"videos": 1, "decoded": 0, "sampled": 0, "decode": 0.0, "convert": 0.0, "letterbox": 0.0, "slot_wait": 0.0}
    max_sec = -1
    error = None
    container = None

    def get_slot():
        if not free:
            t = perf()
            while not free:
                m = conn.recv()
                if m[0] == "release":
                    free.extend(m[1])
                elif m[0] == "stop":
                    raise _Stop
            st["slot_wait"] += perf() - t
        return free.popleft()

    try:
        try:
            container = av.open(io.BytesIO(data), mode="r")
        except Exception as e:
            error = f"could not open video ({short_err(e)})"
            return
        if not container.streams.video:
            error = "no video stream"
            return
        stream = container.streams.video[0]
        cc = stream.codec_context
        if decode_threads > 1:
            cc.thread_type = "AUTO"
        cc.thread_count = decode_threads
        tb = stream.time_base
        rate = stream.average_rate or stream.guessed_rate or stream.base_rate
        num = den = None
        step = 1  # one frame duration in time-base ticks, only used when a frame has no pts
        pts0 = last_pts = None
        seen_slots: set[int] = set()
        n_bad = n_packets = 0
        limited = False  # --max-seconds reached (quick tests only)
        expected_packets = stream.frames or 0  # sample count from the container index (0 if unknown)
        packets = container.demux(stream)
        while True:
            t0 = perf()
            try:
                packet = next(packets)
            except StopIteration:
                break
            except Exception as e:  # the container cannot be read any further
                error = f"demux error after {st['decoded']} frames ({short_err(e)})"
                break
            if packet.size:
                n_packets += 1
            try:
                frames = packet.decode()
            except Exception as e:
                st["decode"] += perf() - t0
                n_bad += 1
                if n_bad > MAX_PACKET_ERRORS:
                    error = f"too many decode errors after {st['decoded']} frames ({short_err(e)})"
                    break
                continue
            st["decode"] += perf() - t0
            for frame in frames:
                if num is None:  # time base of the first frame (stream time base if present)
                    ftb = tb or frame.time_base or Fraction(1, 90000)
                    num, den = ftb.numerator, ftb.denominator
                    if rate:
                        step = max(1, round(Fraction(den, num) / Fraction(rate)))
                pts = frame.pts
                if pts is None:
                    pts = 0 if last_pts is None else last_pts + step
                last_pts = pts
                if pts0 is None:
                    pts0 = pts  # seconds are counted from the first decoded frame
                rel = pts - pts0
                if rel < 0:
                    rel = 0
                sec = rel * num // den  # exact integer arithmetic on pts * time_base
                if sec > MAX_VIDEO_SECONDS:
                    error = error or "frame timestamps out of range"
                    continue
                if max_seconds and sec >= max_seconds:
                    limited = True
                    break
                if sec > max_sec:
                    max_sec = sec
                st["decoded"] += 1
                if sample_fps:  # first frame of each 1/sample_fps slot -> evenly spaced, >= 1 per second
                    g = rel * num * sample_fps // den
                    if g in seen_slots:
                        continue
                    seen_slots.add(g)
                if crash_key == key and st["sampled"] >= 3:
                    os.kill(os.getpid(), signal.SIGSEGV)
                if hang_key == key and st["sampled"] >= 3:
                    time.sleep(10**6)
                t1 = perf()
                img = frame.to_ndarray(format="bgr24")  # only sampled frames are converted
                op = rotation_op(frame_display_matrix(frame))
                t2 = perf()
                img = apply_rotation(img, op, cv2)
                h, w = img.shape[:2]
                new_unpad, top, bottom, left, right, oh, ow = letterbox_params(h, w)
                if (w, h) != new_unpad:
                    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
                img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
                t3 = perf()
                slot = get_slot()
                off = slot * SLOT_BYTES
                buf[off : off + oh * ow * 3] = img.reshape(-1)
                conn.send(("frame", key, sec, slot, oh, ow))
                while conn.poll():  # pick up slot releases early so the pipe never backs up
                    m = conn.recv()
                    if m[0] == "release":
                        free.extend(m[1])
                    elif m[0] == "stop":
                        raise _Stop
                st["sampled"] += 1
                st["convert"] += t2 - t1
                st["letterbox"] += t3 - t2
            if limited:
                break
        if error is None and not limited and expected_packets and n_packets < expected_packets:
            error = f"video data ends early (truncated?): {n_packets} of {expected_packets} packets"
    except _Stop:
        raise
    except Exception as e:
        error = f"decode failed after {st['decoded']} frames ({short_err(e)})"
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
        if max_sec < 0 and error is None:
            error = "no frames decoded"
        conn.send(("done", key, max_sec, error, st))


# --------------------------------------------------------------------------------------------- per-second rules
def aggregate_second(frames: list[tuple[int, ...]] | None):
    """Section 5: present if found in >= half of the frames; people = median, rounded down at .5."""
    if not frames:
        return [], 0
    n = len(frames)
    objects = [name for c, name in enumerate(CLASS_NAMES) if 2 * sum(1 for f in frames if f[c] > 0) >= n]
    p = sorted(f[0] for f in frames)
    people = p[n // 2] if n % 2 else (p[n // 2 - 1] + p[n // 2]) // 2
    return objects, people


# --------------------------------------------------------------------------------------------- GPU side
class Activity:
    """Wall-clock time during which at least one GPU thread is working."""

    def __init__(self):
        self.lock = threading.Lock()
        self.n = 0
        self.t0 = 0.0
        self.busy = 0.0
        self.first = None

    def start(self):
        with self.lock:
            if self.n == 0:
                self.t0 = perf()
                if self.first is None:
                    self.first = self.t0
            self.n += 1

    def stop(self):
        with self.lock:
            self.n -= 1
            if self.n == 0:
                self.busy += perf() - self.t0


class GPUThread(threading.Thread):
    def __init__(self, runner, tid, device):
        super().__init__(name=f"gpu{tid}", daemon=True)
        self.r, self.tid, self.device = runner, tid, device
        self.stats = {"batches": 0, "frames": 0, "padded": 0, "wait": 0.0, "copy": 0.0, "infer": 0.0, "load": 0.0}
        self.ready = threading.Event()
        self.failed = None

    def run(self):
        r = self.r
        try:
            t = perf()
            self.setup()
            self.stats["load"] = perf() - t
        except BaseException as e:
            self.failed = e
            log(f"GPU thread {self.tid} on {self.device} failed to start: {short_err(e)}")
            traceback.print_exc()
            self.ready.set()
            r.wake()
            return
        self.ready.set()
        r.wake()
        while True:
            t = perf()
            with r.idle_lock:
                r.idle_gpu += 1
            r.wake()
            item = r.batch_q.get()
            with r.idle_lock:
                r.idle_gpu -= 1
            self.stats["wait"] += perf() - t
            if item is None:
                return
            (h, w), entries = item
            r.activity.start()
            try:
                counts = self.process(h, w, entries)
            except BaseException as e:
                log(f"inference failed on a batch of {len(entries)}: {short_err(e)}")
                traceback.print_exc()
                counts = None
            finally:
                r.activity.stop()
            r.result_q.put((entries, counts))
            r.wake()

    def setup(self):
        import torch
        from ultralytics.nn.autobackend import AutoBackend
        from ultralytics.utils.nms import non_max_suppression

        self.torch, self.nms = torch, non_max_suppression
        self.cuda = self.device.startswith("cuda")
        dev = torch.device(self.device)
        if self.cuda:
            torch.cuda.set_device(dev)
            torch.backends.cudnn.benchmark = self.r.args.cudnn_benchmark
            self.stream = torch.cuda.Stream(dev)
        self.fp16 = self.cuda and not self.r.args.fp32
        # The same loader the Ultralytics predictor uses: fused Conv+BN, FP16 weights on CUDA.
        self.model = AutoBackend(model=self.r.args.weights, device=dev, fp16=self.fp16, fuse=True, verbose=False)
        self.model.eval()
        self.dev = dev
        b = self.r.args.batch
        self.host = torch.empty(b * SLOT_BYTES, dtype=torch.uint8, pin_memory=self.cuda)
        self.class_idx = torch.tensor(CLASS_IDS, device=dev)
        if self.cuda:  # warm up kernels and NMS
            with torch.inference_mode(), torch.cuda.stream(self.stream):
                x = torch.zeros(1, 3, 384, 640, device=dev, dtype=torch.half if self.fp16 else torch.float)
                self.nms(self.model(x), CONF, IOU, None, False, max_det=MAX_DET, nc=0)
            self.stream.synchronize()

    def padded_size(self, n):
        if not (self.cuda and self.r.args.cudnn_benchmark):
            return n
        b, p = self.r.args.batch, 1
        while p < n:
            p *= 2
        return min(p, b) if n <= b else n  # a few fixed batch sizes -> cudnn benchmarks each only once

    def process(self, h, w, entries):
        torch = self.torch
        t0 = perf()
        n = len(entries)
        host = self.host[: n * h * w * 3].view(n, h, w, 3)
        hn = host.numpy()
        buf = self.r.buf
        size = h * w * 3
        for i, e in enumerate(entries):
            off = e[0] * SLOT_BYTES
            hn[i] = buf[off : off + size].reshape(h, w, 3)
        self.r.release_q.put([e[0] for e in entries])  # slots can be reused now
        self.r.wake()
        t1 = perf()
        counts = self.infer_split(host, n)
        t2 = perf()
        s = self.stats
        s["batches"] += 1
        s["frames"] += n
        s["copy"] += t1 - t0
        s["infer"] += t2 - t1
        return counts

    def infer_split(self, host, n):
        try:
            return self.infer(host)
        except self.torch.cuda.OutOfMemoryError:
            if n == 1:
                raise
            self.torch.cuda.empty_cache()
            log(f"CUDA out of memory with a batch of {n}; splitting")
            k = n // 2
            return self.infer_split(host[:k], k) + self.infer_split(host[k:], n - k)

    def infer(self, host):
        torch = self.torch
        n = host.shape[0]
        npad = self.padded_size(n) - n
        self.stats["padded"] += npad
        ctx = torch.cuda.stream(self.stream) if self.cuda else _null_ctx()
        with torch.inference_mode(), ctx:
            x = host.to(self.dev, non_blocking=True)
            if npad:
                x = torch.cat((x, x.new_zeros((npad,) + tuple(x.shape[1:]))))
            # identical to BasePredictor.preprocess: BHWC->BCHW, contiguous, BGR->RGB, to fp16/fp32, /255
            x = x.permute(0, 3, 1, 2).contiguous()
            x = x.flip(1)
            x = (x.half() if self.fp16 else x.float()).div_(255)
            preds = self.model(x)
            # identical to DetectionPredictor.postprocess (all 80 classes, class-aware NMS)
            dets = self.nms(preds, CONF, IOU, None, False, max_det=MAX_DET, nc=0, end2end=False, rotated=False)
            rows = [torch.bincount(d[:, 5].long(), minlength=80)[self.class_idx] for d in dets[:n]]
            out = torch.stack(rows).cpu().tolist()
        if self.cuda:
            self.stream.synchronize()
        return [tuple(c) for c in out]


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# --------------------------------------------------------------------------------------------- main process
class Video:
    __slots__ = ("key", "dets", "pending", "done", "max_sec", "error")

    def __init__(self, key):
        self.key = key
        self.dets: dict[int, list] = {}
        self.pending = 0
        self.done = False
        self.max_sec = -1
        self.error = None


class Worker:
    def __init__(self, wid, slots):
        self.wid, self.slots = wid, slots
        self.proc = self.conn = None
        self.key = None
        self.last_msg = 0.0
        self.inflight = 0


class Runner:
    def __init__(self, args, files):
        self.args = args
        self.files = files  # list of (idx, path)
        self.ctx = mp.get_context("spawn")
        self.videos: dict[str, Video] = {}
        self.backlog: deque = deque()
        self.buckets: dict[tuple[int, int], list] = {}
        self.batch_q: queue.Queue = queue.Queue()
        self.result_q: queue.Queue = queue.Queue()
        self.release_q: queue.Queue = queue.Queue()
        self.idle_lock = threading.Lock()
        self.idle_gpu = 0
        self.activity = Activity()
        self.wstats = {k: 0.0 for k in ("videos", "decoded", "sampled", "decode", "convert", "letterbox", "slot_wait")}
        self.counts = {"lines": 0, "videos_ok": 0, "videos_err": 0, "worker_restarts": 0, "reader_restarts": 0}
        self.reader_done = False
        self.cur_file = None  # [idx, name, nrows, rows_seen]
        self.next_file_pos = 0  # position in self.files of the first file the reader has not finished
        self.max_backlog = max(4, 2 * args.workers)
        self.last_progress = perf()

    # ------------------------------------------------------------------ setup / teardown
    def wake(self):
        try:
            os.write(self.wake_w, b"x")
        except (BlockingIOError, OSError):
            pass

    def setup(self):
        a = self.args
        nw = a.workers
        need = nw * a.slots_per_worker * SLOT_BYTES
        shm_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None
        if shm_dir:
            st = os.statvfs(shm_dir)
            if st.f_bavail * st.f_frsize < need * 1.25:
                log(f"/dev/shm too small for {need / 2**20:.0f} MiB of frame slots; using a temp file instead")
                shm_dir = None
        fd, self.shm_path = tempfile.mkstemp(prefix="alethia_", dir=shm_dir or tempfile.gettempdir())
        os.ftruncate(fd, need)
        os.close(fd)
        self.shm_bytes = need
        self.buf = np.memmap(self.shm_path, dtype=np.uint8, mode="r+", shape=(need,))
        self.wake_r, self.wake_w = os.pipe()
        os.set_blocking(self.wake_r, False)
        os.set_blocking(self.wake_w, False)
        self.slot_inflight = bytearray(nw * a.slots_per_worker)
        self.workers = [Worker(i, range(i * a.slots_per_worker, (i + 1) * a.slots_per_worker)) for i in range(nw)]
        self.t_start = perf()
        self.start_reader(0)
        for w in self.workers:
            self.start_worker(w)
        # GPU threads load the model while the workers already decode
        self.gpu_threads = [GPUThread(self, i * a.threads_per_device + j, d)
                            for i, d in enumerate(a.devices) for j in range(a.threads_per_device)]
        if a.devices[0].startswith("cuda"):
            import torch

            torch.cuda.init()
        for g in self.gpu_threads:
            g.start()

    def start_reader(self, pos):
        r, wconn = self.ctx.Pipe(duplex=False)
        p = self.ctx.Process(target=reader_main, args=(self.files[pos:], wconn), daemon=True, name="reader")
        p.start()
        wconn.close()
        self.reader_proc, self.reader_conn = p, r

    def start_worker(self, w):
        a = self.args
        free = [s for s in w.slots if not self.slot_inflight[s]]
        mine, theirs = self.ctx.Pipe(duplex=True)
        p = self.ctx.Process(target=worker_main, daemon=True, name=f"decode{w.wid}",
                             args=(w.wid, theirs, self.shm_path, self.shm_bytes, free, a.sample_fps, a.decode_threads,
                                   a.max_seconds))
        p.start()
        theirs.close()
        w.proc, w.conn, w.key, w.last_msg = p, mine, None, perf()

    def shutdown(self):
        for w in getattr(self, "workers", []):
            if w.conn is not None:
                try:
                    w.conn.send(("stop",))
                except Exception:
                    pass
        for _ in getattr(self, "gpu_threads", []):
            self.batch_q.put(None)
        deadline = perf() + 5
        procs = [w.proc for w in getattr(self, "workers", []) if w.proc is not None]
        if getattr(self, "reader_proc", None) is not None:
            procs.append(self.reader_proc)
        for p in procs:
            p.join(timeout=max(0.0, deadline - perf()))
            if p.is_alive():
                p.kill()
                p.join(timeout=2)
        for g in getattr(self, "gpu_threads", []):
            g.join(timeout=10)
        try:
            del self.buf
        except AttributeError:
            pass
        if getattr(self, "shm_path", None):
            try:
                os.unlink(self.shm_path)
            except OSError:
                pass

    # ------------------------------------------------------------------ output
    def write(self, obj):
        self.out.write(json.dumps(obj) + "\n")
        self.counts["lines"] += 1

    def write_error(self, key, msg):
        self.write({"video": key, "error": msg})

    def finalize(self, v: Video):
        del self.videos[v.key]
        last = max(v.max_sec, max(v.dets) if v.dets else -1)
        for s in range(last + 1):
            objects, people = aggregate_second(v.dets.get(s))
            self.write({"video": v.key, "second": s, "objects": objects, "people": people})
        if v.error or last < 0:
            self.write_error(v.key, v.error or "no frames decoded")
            self.counts["videos_err"] += 1
        else:
            self.counts["videos_ok"] += 1

    def maybe_finalize(self, v: Video):
        if v.done and v.pending == 0 and v.key in self.videos:
            self.finalize(v)

    # ------------------------------------------------------------------ message handling
    def on_reader_msg(self, m):
        kind = m[0]
        if kind == "video":
            self.cur_file[3] += 1
            if m[2] is None:
                self.write_error(m[1], "no video data (null or empty mp4)")
                self.counts["videos_err"] += 1
            else:
                self.backlog.append((m[1], m[2]))
        elif kind == "video_error":
            self.cur_file[3] += 1
            self.write_error(m[1], m[2])
            self.counts["videos_err"] += 1
        elif kind == "file":
            self.cur_file = [m[1], m[2], m[3], 0]
        elif kind == "file_end":
            self.cur_file = None
            self.next_file_pos = self.pos_of(m[1]) + 1
        elif kind == "file_error":
            self.write_error(m[2], m[3])
            self.counts["videos_err"] += 1
            self.next_file_pos = self.pos_of(m[1]) + 1
        elif kind == "eof":
            self.reader_done = True

    def pos_of(self, idx):
        for p, (i, _) in enumerate(self.files):
            if i == idx:
                return p
        return len(self.files)

    def on_reader_died(self):
        try:
            self.reader_proc.join(timeout=1)
        except Exception:
            pass
        code = self.reader_proc.exitcode
        self.reader_conn.close()
        self.reader_conn = None
        if self.cur_file is not None:
            idx, name, nrows, seen = self.cur_file
            for r in range(seen, nrows):
                self.write_error(f"{name}:{r}", f"parquet reader crashed (exit code {code})")
                self.counts["videos_err"] += 1
            self.next_file_pos = self.pos_of(idx) + 1
            self.cur_file = None
        elif self.next_file_pos < len(self.files):
            name = os.path.basename(self.files[self.next_file_pos][1])
            self.write_error(name, f"parquet reader crashed while opening the file (exit code {code})")
            self.counts["videos_err"] += 1
            self.next_file_pos += 1
        self.counts["reader_restarts"] += 1
        if self.next_file_pos < len(self.files) and self.counts["reader_restarts"] <= len(self.files) + 3:
            log(f"reader process died (exit code {code}); restarting at file {self.next_file_pos}")
            self.start_reader(self.next_file_pos)
        else:
            self.reader_done = True

    def on_worker_msg(self, w: Worker, m):
        kind = m[0]
        w.last_msg = perf()
        if kind == "frame":
            _, key, sec, slot, h, wd = m
            v = self.videos.get(key)
            self.slot_inflight[slot] = 1
            w.inflight += 1
            if v is None:  # should not happen; give the slot straight back
                self.release([slot])
                return
            v.pending += 1
            if sec > v.max_sec:
                v.max_sec = sec
            self.buckets.setdefault((h, wd), []).append((slot, key, sec))
        elif kind == "done":
            _, key, max_sec, err, st = m
            for k, val in st.items():
                self.wstats[k] = self.wstats.get(k, 0) + val
            w.key = None
            v = self.videos.get(key)
            if v is not None:
                v.done = True
                v.max_sec = max(v.max_sec, max_sec)
                v.error = v.error or err
                self.maybe_finalize(v)

    def on_worker_died(self, w: Worker):
        try:
            w.proc.join(timeout=2)
        except Exception:
            pass
        if w.proc.is_alive():
            w.proc.kill()
            w.proc.join(timeout=2)
        code = w.proc.exitcode
        try:
            w.conn.close()
        except Exception:
            pass
        w.conn = None
        if w.key is not None:
            v = self.videos.get(w.key)
            if v is not None:
                v.done = True
                v.error = v.error or f"decoder process crashed (exit code {code})"
                self.maybe_finalize(v)
            log(f"decode worker {w.wid} died (exit code {code}) while on {w.key}; restarting it")
        w.key = None
        self.counts["worker_restarts"] += 1
        if self.counts["worker_restarts"] <= 100 + 10 * len(self.workers):
            self.start_worker(w)
        else:
            w.proc = None

    def release(self, slots):
        by_worker: dict[int, list] = {}
        spw = self.args.slots_per_worker
        for s in slots:
            self.slot_inflight[s] = 0
            by_worker.setdefault(s // spw, []).append(s)
        for wid, ss in by_worker.items():
            w = self.workers[wid]
            w.inflight -= len(ss)
            if w.conn is not None:
                try:
                    w.conn.send(("release", ss))
                except (BrokenPipeError, OSError):
                    pass  # death is noticed through the pipe EOF

    def on_results(self, entries, counts):
        for i, (slot, key, sec) in enumerate(entries):
            v = self.videos.get(key)
            if v is None:
                continue
            v.pending -= 1
            if counts is None:
                v.error = v.error or "inference failed"
            else:
                v.dets.setdefault(sec, []).append(counts[i])
            self.maybe_finalize(v)

    # ------------------------------------------------------------------ scheduling
    def dispatch(self):
        for w in self.workers:
            if not self.backlog:
                return
            if w.key is None and w.conn is not None:
                key, data = self.backlog.popleft()
                self.videos[key] = Video(key)
                w.key, w.last_msg = key, perf()
                try:
                    w.conn.send(("task", key, data))
                except (BrokenPipeError, OSError):
                    pass  # handled as a crash on this video when the pipe reports EOF

    def schedule(self):
        b = self.args.batch
        for shape, lst in self.buckets.items():
            while len(lst) >= b:
                self.batch_q.put((shape, lst[:b]))
                del lst[:b]
        for s in [s for s, lst in self.buckets.items() if not lst]:
            del self.buckets[s]
        # work-conserving: while a GPU thread would otherwise sit idle, send the fullest partial batch
        while self.buckets and self.batch_q.qsize() < self.idle_gpu:
            shape = max(self.buckets, key=lambda s: len(self.buckets[s]))
            self.batch_q.put((shape, self.buckets.pop(shape)))

    def watchdog(self):
        now = perf()
        for w in self.workers:
            if w.key is not None and w.conn is not None and w.inflight < len(w.slots):
                if now - w.last_msg > self.args.video_timeout:
                    log(f"decode worker {w.wid} silent for {now - w.last_msg:.0f}s on {w.key}; killing it")
                    v = self.videos.get(w.key)
                    if v is not None:
                        v.error = f"decoding timed out after {self.args.video_timeout:.0f}s without progress"
                    w.proc.kill()
                    w.last_msg = now
        if self.gpu_threads and all(g.failed is not None for g in self.gpu_threads):
            raise RuntimeError("no GPU thread could load the model")

    def progress(self, force=False):
        now = perf()
        if force or now - self.last_progress > self.args.progress_every:
            self.last_progress = now
            el = now - self.t_start
            fr = sum(g.stats["frames"] for g in self.gpu_threads)
            c = self.counts
            log(f"{el:7.0f}s | videos done {c['videos_ok']} ok / {c['videos_err']} err, active {len(self.videos)}, "
                f"queued {len(self.backlog)} | frames inferred {fr} ({fr / max(el, 1e-9):.1f}/s) | "
                f"decoded {int(self.wstats['decoded'])}")

    # ------------------------------------------------------------------ main loop
    def finished(self):
        return (self.reader_done and not self.backlog and not self.videos
                and all(w.key is None for w in self.workers))

    def loop(self):
        while not self.finished():
            watch = [self.wake_r]
            conn_owner = {}
            for w in self.workers:
                if w.conn is not None:
                    watch.append(w.conn)
                    conn_owner[w.conn] = w
            if self.reader_conn is not None and not self.reader_done and len(self.backlog) < self.max_backlog:
                watch.append(self.reader_conn)
            ready = mp_wait(watch, timeout=1.0)
            for c in ready:
                if c == self.wake_r:
                    try:
                        while os.read(self.wake_r, 65536):
                            pass
                    except BlockingIOError:
                        pass
                elif c is self.reader_conn:
                    try:
                        for _ in range(64):
                            self.on_reader_msg(self.reader_conn.recv())
                            if self.reader_done or len(self.backlog) >= self.max_backlog or not self.reader_conn.poll():
                                break
                    except (EOFError, OSError):
                        if not self.reader_done:
                            self.on_reader_died()
                        else:
                            self.reader_conn = None
                else:
                    w = conn_owner[c]
                    try:
                        for _ in range(512):
                            self.on_worker_msg(w, w.conn.recv())
                            if not w.conn.poll():
                                break
                    except (EOFError, OSError):
                        self.on_worker_died(w)
            while True:
                try:
                    self.release(self.release_q.get_nowait())
                except queue.Empty:
                    break
            while True:
                try:
                    self.on_results(*self.result_q.get_nowait())
                except queue.Empty:
                    break
            self.dispatch()
            self.schedule()
            self.watchdog()
            self.progress()

    def run(self):
        out_dir = os.path.dirname(os.path.abspath(self.args.output))
        os.makedirs(out_dir, exist_ok=True)
        with open(self.args.output, "w", encoding="utf-8", buffering=1 << 20) as self.out:
            try:
                self.setup()
                self.loop()
            finally:
                self.out.flush()
                self.shutdown()
        self.t_end = perf()
        self.summary()

    def summary(self):
        a, ws, c = self.args, self.wstats, self.counts
        wall = self.t_end - self.t_start
        gs = [g.stats for g in self.gpu_threads]
        frames = sum(s["frames"] for s in gs)
        act = self.activity
        first = (act.first - self.t_start) if act.first else float("nan")
        gpu_window = wall - first if act.first else 0.0
        starved = max(0.0, gpu_window - act.busy)
        nw = max(1, a.workers)
        lines = [
            "",
            "================ timing breakdown ================",
            f"wall time                    {wall:10.1f} s",
            f"videos                       {c['videos_ok']} ok, {c['videos_err']} with error line"
            f"  (worker restarts {c['worker_restarts']}, reader restarts {c['reader_restarts']})",
            f"frames decoded / sampled     {int(ws['decoded'])} / {int(ws['sampled'])}"
            f"   -> {frames} inferred ({frames / max(wall, 1e-9):.1f} frames/s overall)",
            f"output lines                 {c['lines']}",
            "-- GPU (main process) --",
            f"devices                      {', '.join(a.devices)} x {a.threads_per_device} thread(s), batch {a.batch}, "
            f"{'fp32' if a.fp32 or not a.devices[0].startswith('cuda') else 'fp16'}",
            f"model load (max over thr.)   {max((s['load'] for s in gs), default=0):10.1f} s",
            f"first batch started at       {first:10.1f} s",
            f"GPU busy (>=1 thread)        {act.busy:10.1f} s   ({100 * act.busy / max(gpu_window, 1e-9):5.1f}% of the time after the first batch)",
            f"GPU starved = decode wait    {starved:10.1f} s   ({100 * starved / max(gpu_window, 1e-9):5.1f}%)",
            f"  per-thread sums: copy {sum(s['copy'] for s in gs):.1f} s, forward+NMS {sum(s['infer'] for s in gs):.1f} s, "
            f"waiting for batches {sum(s['wait'] for s in gs):.1f} s",
            f"  batches {sum(s['batches'] for s in gs)}, mean size {frames / max(1, sum(s['batches'] for s in gs)):.1f}, "
            f"padding frames {sum(s['padded'] for s in gs)}",
            f"-- decode workers ({nw} processes, sums; per worker = sum / {nw}) --",
            f"decode (demux+decode)        {ws['decode']:10.1f} s   ({ws['decode'] / nw:.1f} s per worker)",
            f"to BGR (sampled only)        {ws['convert']:10.1f} s   ({ws['convert'] / nw:.1f} s per worker)",
            f"rotate + letterbox           {ws['letterbox']:10.1f} s   ({ws['letterbox'] / nw:.1f} s per worker)",
            f"blocked on free slots (GPU)  {ws['slot_wait']:10.1f} s   ({ws['slot_wait'] / nw:.1f} s per worker)",
            "==================================================",
        ]
        print("\n".join(lines), file=sys.stderr, flush=True)
        if a.timing_json:
            gsum = lambda k: round(sum(s[k] for s in gs), 3)
            data = {
                "wall_s": round(wall, 3), "videos_ok": c["videos_ok"], "videos_err": c["videos_err"],
                "worker_restarts": c["worker_restarts"], "reader_restarts": c["reader_restarts"],
                "frames_decoded": int(ws["decoded"]), "frames_sampled": int(ws["sampled"]), "frames_inferred": frames,
                "output_lines": c["lines"], "devices": a.devices, "threads_per_device": a.threads_per_device,
                "batch": a.batch, "sample_fps": a.sample_fps, "workers": a.workers,
                "precision": "fp32" if a.fp32 or not a.devices[0].startswith("cuda") else "fp16",
                "model_load_s": round(max((s["load"] for s in gs), default=0), 3),
                "first_batch_at_s": round(first, 3) if act.first else None,
                "gpu_busy_s": round(act.busy, 3), "gpu_starved_s": round(starved, 3),
                "gpu_thread_copy_s": gsum("copy"), "gpu_thread_infer_s": gsum("infer"),
                "gpu_thread_wait_s": gsum("wait"), "batches": int(gsum("batches")), "padded_frames": int(gsum("padded")),
                "worker_decode_s": round(ws["decode"], 3), "worker_convert_s": round(ws["convert"], 3),
                "worker_letterbox_s": round(ws["letterbox"], 3), "worker_slot_wait_s": round(ws["slot_wait"], 3),
            }
            with open(a.timing_json, "w") as f:
                json.dump(data, f, indent=1)


# --------------------------------------------------------------------------------------------- CLI
def list_parquet(path):
    if os.path.isfile(path):
        return [path]
    out = []
    for root, _, names in os.walk(path):
        out += [os.path.join(root, n) for n in names if n.endswith(".parquet")]
    return sorted(out, key=lambda p: (os.path.basename(p), p))


def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        ncpu = len(os.sched_getaffinity(0))
    except AttributeError:
        ncpu = os.cpu_count() or 2
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", required=True, help="folder of .parquet files (searched recursively) or one file")
    p.add_argument("--output", required=True, help="output .jsonl path")
    p.add_argument("--weights", default=os.path.join(here, "yolo11l.pt"))
    p.add_argument("--device", default="auto", help="'auto' (all GPUs, else CPU), 'cpu', '0', or '0,1'")
    p.add_argument("--batch", type=int, default=32, help="max frames per GPU batch")
    p.add_argument("--workers", type=int, default=max(1, ncpu - 1), help="decode processes (default: CPUs - 1)")
    p.add_argument("--decode-threads", type=int, default=1, help="FFmpeg threads per decode process")
    p.add_argument("--sample-fps", type=int, default=5, help="frames sampled per second of video; 0 = every frame")
    p.add_argument("--threads-per-device", type=int, default=None, help="GPU threads (model copies) per device")
    p.add_argument("--slots-per-worker", type=int, default=None, help="shared-memory frame slots per worker")
    p.add_argument("--fp32", action="store_true", help="disable FP16 on CUDA")
    p.add_argument("--no-cudnn-benchmark", dest="cudnn_benchmark", action="store_false")
    p.add_argument("--video-timeout", type=float, default=300.0, help="kill a decoder silent this long (s)")
    p.add_argument("--progress-every", type=float, default=60.0, help="progress log interval (s)")
    p.add_argument("--timing-json", default=None, help="also write the timing breakdown to this JSON file")
    p.add_argument("--max-seconds", type=int, default=0, help=argparse.SUPPRESS)  # quick tests: first N s only
    a = p.parse_args(argv)
    if a.device == "auto" or a.device != "cpu":
        import torch

        if a.device == "auto":
            n = torch.cuda.device_count()
            a.devices = [f"cuda:{i}" for i in range(n)] or ["cpu"]
        else:
            a.devices = [f"cuda:{d.strip()}" for d in a.device.split(",") if d.strip() != ""]
            if not torch.cuda.is_available():
                p.error("CUDA requested but torch.cuda.is_available() is False")
    else:
        a.devices = ["cpu"]
    if a.threads_per_device is None:
        a.threads_per_device = 2 if a.devices[0].startswith("cuda") else 1
    if a.slots_per_worker is None:  # ~1.5 GiB of slots in total, 16..64 per worker
        a.slots_per_worker = int(min(64, max(16, (1536 << 20) // (a.workers * SLOT_BYTES))))
    a.batch = max(1, a.batch)
    a.sample_fps = max(0, a.sample_fps)
    return a


def main(argv=None):
    os.environ.setdefault("YOLO_OFFLINE", "1")  # never let Ultralytics touch the network during the run
    os.environ.setdefault("YOLO_VERBOSE", "False")
    args = parse_args(argv)
    files = list(enumerate(list_parquet(args.input)))
    if not os.path.isfile(args.weights):
        sys.exit(f"weights not found: {args.weights} (run: python prepare.py)")
    log(f"{len(files)} parquet file(s), {args.workers} decode workers, devices {args.devices} "
        f"x{args.threads_per_device}, batch {args.batch}, sample_fps {args.sample_fps or 'all'}")

    def on_term(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, on_term)
    runner = Runner(args, files)
    if not files:
        open(args.output, "w").close()
        log("no parquet files found; wrote an empty output")
        return 0
    runner.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
