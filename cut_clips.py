import argparse
import glob
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from src.court.court_line_detector import CourtLineDetector


def _segments(flags, max_gap, min_len, max_len):
    raw = []
    n = len(flags)
    i = 0
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i + 1
        while j < n and flags[j]:
            j += 1
        raw.append([i, j])
        i = j
    if not raw:
        return []
    merged = [raw[0]]
    for a, b in raw[1:]:
        if a - merged[-1][1] <= max_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        if b - a < min_len:
            continue
        s = a
        while b - s > max_len:
            out.append((s, s + max_len))
            s += max_len
        if b - s >= min_len:
            out.append((s, b))
        elif out:
            ps, _ = out[-1]
            out[-1] = (ps, b)
    return out


def _scan(path, court, max_frames):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)
    flags = []
    prev_gray = None
    prev_det = None
    with tqdm(total=total, desc=os.path.basename(path)) as bar:
        n = 0
        while n < total:
            ok, frame = cap.read()
            if not ok:
                break
            try:
                gray, det = court.track_frame(frame, n, prev_gray, prev_det)
                use = bool(det.quality_ok)
            except RuntimeError:
                gray, det, use = None, None, False
            if use:
                prev_gray, prev_det = gray, det
            else:
                prev_gray, prev_det = None, None
            flags.append(use)
            n += 1
            bar.update(1)
    cap.release()
    return flags, fps, w, h


def _write(path, out_path, start, end, fps, w, h):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(end - start):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
    writer.release()
    cap.release()


def _cut(path, out_root, court, max_frames, min_sec, max_sec, gap_sec):
    stem = os.path.splitext(os.path.basename(path))[0]
    flags, fps, w, h = _scan(path, court, max_frames)
    segs = _segments(
        flags,
        max_gap=max(1, int(round(gap_sec * fps))),
        min_len=max(1, int(round(min_sec * fps))),
        max_len=max(2, int(round(max_sec * fps))),
    )
    dest = os.path.join(out_root, stem)
    clips = []
    for i, (a, b) in enumerate(segs):
        name = f"clip_{i:04d}.mp4"
        _write(path, os.path.join(dest, name), a, b, fps, w, h)
        clips.append({"file": name, "start": int(a), "end": int(b)})
        print(stem, name, a, b, "sec", round((b - a) / fps, 1))
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"video": os.path.abspath(path), "fps": fps, "clips": clips}, f, ensure_ascii=False, indent=2)
    kept = int(np.sum(flags))
    print(stem, "clips", len(clips), "court_frames", kept, "/", len(flags))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="high_angle_dataset")
    p.add_argument("--out", default="high_angle_dataset/clips")
    p.add_argument("--only", default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--min-sec", type=float, default=2.0)
    p.add_argument("--max-sec", type=float, default=20.0)
    p.add_argument("--gap-sec", type=float, default=0.4)
    p.add_argument("--heatmap-weights", default="weights/court_heatmap.pth")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.videos):
        raise SystemExit(f"videos not found: {args.videos}")
    paths = sorted(glob.glob(os.path.join(args.videos, "*.mp4")))
    paths = [v for v in paths if "tennis_high_angle_01" not in os.path.basename(v)]
    if args.only:
        paths = [v for v in paths if args.only in os.path.basename(v)]
    if not paths:
        raise SystemExit(f"no mp4 in {args.videos}")
    court = CourtLineDetector(heatmap_path=args.heatmap_weights)
    os.makedirs(args.out, exist_ok=True)
    for path in paths:
        _cut(path, args.out, court, args.max_frames, args.min_sec, args.max_sec, args.gap_sec)


if __name__ == "__main__":
    main()
