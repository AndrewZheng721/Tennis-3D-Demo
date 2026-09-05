import argparse
import csv
import glob
import os
import random
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from src.ball.tracknet_tracker import TrackNetBallTracker, default_ball_weights
from src.court.court_line_detector import CourtLineDetector


def _find_high_angle_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [
        os.path.join(here, "high_angle_dataset"),
        os.path.join(here, "..", "训练数据集-网球视频", "high_angle"),
        os.path.join(os.path.dirname(here), "训练数据集-网球视频", "high_angle"),
    ]
    for r in roots:
        if os.path.isdir(r):
            return os.path.abspath(r)
    matches = glob.glob(os.path.join(os.path.dirname(here), "*", "high_angle"))
    return matches[0] if matches else ""


def _centers(filled):
    pts = []
    for item in filled:
        box = item.get(1, [])
        if box and len(box) >= 4 and np.isfinite(box[0]):
            pts.append(((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0))
        else:
            pts.append(None)
    return pts


def _good_ranges(pts, min_len=8):
    n = len(pts)
    ranges = []
    i = 0
    while i < n:
        if pts[i] is None:
            i += 1
            continue
        j = i + 1
        while j < n and pts[j] is not None:
            j += 1
        if j - i >= min_len:
            xs = np.array([p[0] for p in pts[i:j]])
            ys = np.array([p[1] for p in pts[i:j]])
            step = np.hypot(np.diff(xs), np.diff(ys))
            path = float(step.sum()) if len(step) else 0.0
            disp = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
            med = float(np.median(step)) if len(step) else 0.0
            if med >= 3.0 and not (path > 80 and disp / max(path, 1.0) < 0.12):
                ranges.append((i, j))
        i = j
    return ranges


def _write_video_labels(path, filled, out_img, stem, orig_w, orig_h, in_w, in_h):
    pts = _centers(filled)
    keep = set()
    rows = []
    for a, b in _good_ranges(pts):
        for i in range(a, b):
            keep.add(i)
            if i >= 2:
                keep.add(i - 1)
                keep.add(i - 2)
    if not keep:
        return rows
    os.makedirs(os.path.join(out_img, stem), exist_ok=True)
    cap = cv2.VideoCapture(path)
    sx, sy = in_w / float(orig_w), in_h / float(orig_h)
    fid = 0
    saved = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fid in keep:
            img = cv2.resize(frame, (in_w, in_h))
            rel = f"images/{stem}/{fid:06d}.jpg"
            cv2.imwrite(os.path.join(out_img, stem, f"{fid:06d}.jpg"), img)
            saved[fid] = rel
        fid += 1
    cap.release()
    for a, b in _good_ranges(pts):
        for i in range(max(a, 2), b):
            if i not in saved or (i - 1) not in saved or (i - 2) not in saved:
                continue
            x, y = pts[i]
            rows.append(
                {
                    "path1": saved[i],
                    "path2": saved[i - 1],
                    "path3": saved[i - 2],
                    "x": round(x * sx, 2),
                    "y": round(y * sy, 2),
                    "vis": 1,
                }
            )
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default=None)
    p.add_argument("--out", default="dataset/tracknet_auto")
    p.add_argument("--weights", default=None)
    p.add_argument("--heatmap-weights", default="weights/court_heatmap.pth")
    p.add_argument("--max-frames", type=int, default=15000)
    p.add_argument("--keep-test", action="store_true")
    p.add_argument("--only", default=None, help="只处理文件名包含该字符串的视频，例如 tennis_high_angle_02")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _load_existing_rows(out_dir, drop_stems):
    rows = []
    fields = ["path1", "path2", "path3", "x", "y", "vis"]
    for name in ("labels_train.csv", "labels_val.csv"):
        path = os.path.join(out_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                blob = row.get("path1", "") + row.get("path2", "") + row.get("path3", "")
                if any(stem in blob for stem in drop_stems):
                    continue
                rows.append({k: row[k] for k in fields})
    return rows


def main():
    args = parse_args()
    video_dir = args.videos or _find_high_angle_dir()
    if not video_dir or not os.path.isdir(video_dir):
        raise FileNotFoundError("找不到 high_angle 目录，用 --videos 指定")
    weights = args.weights or default_ball_weights()[0]
    if not os.path.isfile(weights):
        raise FileNotFoundError(weights)
    videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if not args.keep_test:
        videos = [v for v in videos if "high_angle_01" not in os.path.basename(v)]
    if args.only:
        videos = [v for v in videos if args.only in os.path.basename(v)]
    if not videos:
        raise FileNotFoundError(video_dir)
    if not os.path.isfile(args.heatmap_weights):
        raise FileNotFoundError(args.heatmap_weights)
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    court = CourtLineDetector(heatmap_path=args.heatmap_weights)
    tracker = TrackNetBallTracker(weights)
    drop_stems = [os.path.splitext(os.path.basename(v))[0] for v in videos]
    all_rows = _load_existing_rows(args.out, drop_stems)
    for vp in videos:
        stem = os.path.splitext(os.path.basename(vp))[0]
        img_dir = os.path.join(args.out, "images", stem)
        if os.path.isdir(img_dir):
            shutil.rmtree(img_dir)
        print("label", vp)
        cap = cv2.VideoCapture(vp)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if args.max_frames:
            total = min(total, args.max_frames)
        tracker.reset()
        raw = []
        court_ok = []
        prev_gray = None
        prev_det = None
        with tqdm(total=total, desc=stem) as bar:
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
                    tracker.set_court(det.keypoints_xy)
                    raw.append(tracker.detect_frame(frame))
                else:
                    prev_gray, prev_det = None, None
                    tracker.reset()
                    raw.append({})
                court_ok.append(use)
                n += 1
                bar.update(1)
        cap.release()
        filled = tracker.interpolate_ball_positions(raw)
        for i, use in enumerate(court_ok):
            if not use:
                filled[i] = {}
        rows = _write_video_labels(
            vp, filled, os.path.join(args.out, "images"), stem, w, h, 640, 360
        )
        print("  court_ok", sum(court_ok), "/", len(court_ok), "samples", len(rows))
        all_rows.extend(rows)
    random.Random(args.seed).shuffle(all_rows)
    n_val = max(1, int(len(all_rows) * args.val_ratio))
    val_rows = all_rows[:n_val]
    train_rows = all_rows[n_val:]
    fields = ["path1", "path2", "path3", "x", "y", "vis"]
    for name, rows in (("labels_train.csv", train_rows), ("labels_val.csv", val_rows)):
        with open(os.path.join(args.out, name), "w", newline="", encoding="utf-8") as f:
            wri = csv.DictWriter(f, fieldnames=fields)
            wri.writeheader()
            wri.writerows(rows)
    print("train", len(train_rows), "val", len(val_rows), "->", args.out)


if __name__ == "__main__":
    main()
