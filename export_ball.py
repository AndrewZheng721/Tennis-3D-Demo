import argparse
import csv
import glob
import os
import random

import cv2
import numpy as np


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _load_fixes(path):
    fixes = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fixes[int(row["frame"])] = [float(row["x"]), float(row["y"]), int(row["vis"])]
    return fixes


def _load_events(path):
    marks = {}
    if not os.path.isfile(path):
        return marks
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            marks[int(row["frame"])] = int(row["status"])
    return marks


def _smooth(fixes, marks):
    ids = [i for i, v in fixes.items() if v[2] > 0]
    xs = {i: fixes[i][0] for i in ids}
    ys = {i: fixes[i][1] for i in ids}
    for i in ids:
        if marks.get(i) in (1, 2):
            continue
        nb = [j for j in (i - 1, i, i + 1) if j in xs and marks.get(j) not in (1, 2)]
        if len(nb) < 3:
            continue
        fixes[i][0] = float(np.median([xs[j] for j in nb]))
        fixes[i][1] = float(np.median([ys[j] for j in nb]))


def _grab(cap, want):
    saved = {}
    pos = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for fid in sorted(want):
        if fid < pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            pos = fid
        while pos <= fid:
            ok, frame = cap.read()
            if not ok:
                break
            if pos == fid:
                saved[fid] = cv2.resize(frame, (640, 360))
            pos += 1
    return saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="high_angle_dataset")
    p.add_argument("--only", default=None)
    p.add_argument("--out", default="dataset/tracknet_ball")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    label_dir = os.path.join(args.videos, "labels_ball")
    paths = sorted(glob.glob(os.path.join(label_dir, "*.csv")))
    paths = [x for x in paths if "high_angle_01" not in os.path.basename(x)]
    if args.only:
        paths = [x for x in paths if args.only in os.path.basename(x)]
    if not paths:
        raise SystemExit(f"no csv in {label_dir}")
    rows = []
    for lp in paths:
        stem = _stem(lp)
        video = os.path.join(args.videos, stem + ".mp4")
        if not os.path.isfile(video):
            raise SystemExit(f"missing {video}")
        fixes = _load_fixes(lp)
        _smooth(fixes, _load_events(os.path.join(args.videos, "labels", stem + ".csv")))
        need = set()
        use = []
        for fid, (x, y, vis) in fixes.items():
            if fid < 2:
                continue
            use.append(fid)
            need.update((fid, fid - 1, fid - 2))
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise SystemExit(f"cannot open {video}")
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1
        sx, sy = 640.0 / float(w), 360.0 / float(h)
        img_dir = os.path.join(args.out, "images", stem)
        os.makedirs(img_dir, exist_ok=True)
        saved = _grab(cap, need)
        cap.release()
        n1 = n0 = 0
        for fid in sorted(use):
            if any(k not in saved for k in (fid, fid - 1, fid - 2)):
                continue
            rel = {}
            for k in (fid, fid - 1, fid - 2):
                name = f"{k:06d}.jpg"
                cv2.imwrite(os.path.join(img_dir, name), saved[k])
                rel[k] = f"images/{stem}/{name}"
            x, y, vis = fixes[fid]
            if vis <= 0:
                x = y = 0.0
                n0 += 1
            else:
                x, y = round(x * sx, 2), round(y * sy, 2)
                n1 += 1
            rows.append(
                {
                    "path1": rel[fid],
                    "path2": rel[fid - 1],
                    "path3": rel[fid - 2],
                    "x": x,
                    "y": y,
                    "vis": 1 if vis > 0 else 0,
                }
            )
        print(stem, "ball", n1, "empty", n0)
    if len(rows) < 5:
        raise SystemExit(f"too few samples: {len(rows)}")
    random.Random(args.seed).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_ratio))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    fields = ["path1", "path2", "path3", "x", "y", "vis"]
    os.makedirs(args.out, exist_ok=True)
    for name, part in (("labels_train.csv", train_rows), ("labels_val.csv", val_rows)):
        with open(os.path.join(args.out, name), "w", newline="", encoding="utf-8") as f:
            wri = csv.DictWriter(f, fieldnames=fields)
            wri.writeheader()
            wri.writerows(part)
    print("train", len(train_rows), "val", len(val_rows), "->", args.out)


if __name__ == "__main__":
    main()
