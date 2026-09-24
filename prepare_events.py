import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from src.ball.tracknet_tracker import create_ball_tracker, default_ball_weights
from src.court.court_line_detector import CourtLineDetector
from src.score.bounce import _centers, _geom_bounces
from src.score.geom import ball_center
from src.score.players import PlayerTracker, default_person_weights


def _segments(flags, max_gap=12, min_len=45):
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
    return [(a, b) for a, b in merged if b - a >= min_len]


def _load_marks(video_path):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    path = os.path.join(os.path.dirname(video_path), "labels", stem + ".csv")
    marks = {}
    if not os.path.isfile(path):
        return marks
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            marks[int(row["frame"])] = int(row["status"])
    return marks


def _write_clip(out_dir, filled, players, src_ids, marks=None):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for local_i, src_i in enumerate(src_ids):
        uv = ball_center(filled[src_i].get(1, []))
        if uv is None:
            vis, x, y = 0, 0, 0
        else:
            vis, x, y = 1, int(round(uv[0])), int(round(uv[1]))
        rows.append([f"{local_i:04d}.jpg", vis, x, y, int((marks or {}).get(src_i, 0))])
    with open(os.path.join(out_dir, "Label.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file name", "visibility", "x-coordinate", "y-coordinate", "status"])
        w.writerows(rows)
    pl = []
    for local_i, src_i in enumerate(src_ids):
        pl.append({"frame_id": local_i, "src_frame": int(src_i), "players": players[src_i]})
    with open(os.path.join(out_dir, "players.json"), "w", encoding="utf-8") as f:
        json.dump({"frames": pl}, f, ensure_ascii=False)
    xs, ys = _centers([filled[i] for i in src_ids])
    cands = _geom_bounces(xs, ys, min_gap=6)
    with open(os.path.join(out_dir, "candidates.txt"), "w", encoding="utf-8") as f:
        for i in cands:
            f.write(f"{i}\n")
    return len(rows), len(cands)


def _process_video(path, out_root, court, tracker, players, max_frames):
    marks = _load_marks(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)
    tracker.reset()
    if players is not None:
        players.reset()
    filled_raw = []
    pose = []
    court_ok = []
    prev_gray = None
    prev_det = None
    with tqdm(total=total, desc=stem) as bar:
        n = 0
        while n < total:
            ok, frame = cap.read()
            if not ok:
                break
            if prev_det is None and n % 8 != 0:
                filled_raw.append({})
                pose.append([])
                court_ok.append(False)
                n += 1
                bar.update(1)
                continue
            try:
                gray, det = court.track_frame(frame, n, prev_gray, prev_det, redetect_every=60)
                use = bool(det.quality_ok)
            except RuntimeError:
                gray, det, use = None, None, False
            if use:
                prev_gray, prev_det = gray, det
                tracker.set_court(det.keypoints_xy)
                filled_raw.append(tracker.detect_frame(frame))
                if players is None:
                    pose.append([])
                elif n % 5 == 0 or not pose or not pose[-1]:
                    pose.append(players.detect(frame, det))
                else:
                    pose.append(pose[-1])
            else:
                prev_gray, prev_det = None, None
                tracker.reset()
                if players is not None:
                    players.reset()
                filled_raw.append({})
                pose.append([])
            court_ok.append(use)
            n += 1
            bar.update(1)
    cap.release()
    filled = tracker.interpolate_ball_positions(filled_raw)
    for i, use in enumerate(court_ok):
        if not use:
            filled[i] = {}
    segs = _segments(court_ok)
    nclip = 0
    for a, b in segs:
        ids = list(range(a, b))
        clip_dir = os.path.join(out_root, stem, f"clip_{nclip:04d}")
        nframe, ncand = _write_clip(clip_dir, filled, pose, ids, marks)
        print(stem, f"clip_{nclip:04d}", "frames", a, b, "n", nframe, "candidates", ncand)
        nclip += 1
    print(stem, "clips", nclip, "court_ok", int(np.sum(court_ok)), "/", len(court_ok))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="high_angle_dataset")
    p.add_argument("--out", default="high_angle_dataset/events")
    p.add_argument("--only", default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--heatmap-weights", default="weights/court_heatmap.pth")
    p.add_argument("--ball-weights", default=None)
    p.add_argument("--pose-weights", default=None)
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
    ball_w, _ = default_ball_weights()
    ball_w = args.ball_weights or ball_w
    court = CourtLineDetector(heatmap_path=args.heatmap_weights)
    tracker = create_ball_tracker(ball_w)
    pose_w = args.pose_weights or default_person_weights()
    person = PlayerTracker(pose_w) if pose_w else None
    os.makedirs(args.out, exist_ok=True)
    for path in paths:
        _process_video(path, args.out, court, tracker, person, args.max_frames)


if __name__ == "__main__":
    main()
