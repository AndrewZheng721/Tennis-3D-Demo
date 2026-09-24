import argparse
import csv
import glob
import json
import os

import cv2


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _marks_path(video):
    return os.path.join(os.path.dirname(os.path.abspath(video)), "labels", _stem(video) + ".csv")


def _load_marks(path):
    marks = {}
    if not os.path.isfile(path):
        return marks
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            marks[int(row["frame"])] = int(row["status"])
    return marks


def _load_tracks(video):
    root = os.path.join(os.path.dirname(os.path.abspath(video)), "events", _stem(video))
    ball = {}
    players = {}
    if not os.path.isdir(root):
        return ball, players
    for clip in sorted(glob.glob(os.path.join(root, "clip_*"))):
        pj = os.path.join(clip, "players.json")
        lj = os.path.join(clip, "Label.csv")
        if not (os.path.isfile(pj) and os.path.isfile(lj)):
            continue
        with open(pj, encoding="utf-8") as f:
            frames = json.load(f).get("frames") or []
        with open(lj, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for fr, row in zip(frames, rows):
            src = int(fr["src_frame"])
            if int(row["visibility"]) > 0:
                ball[src] = (int(row["x-coordinate"]), int(row["y-coordinate"]))
            players[src] = fr.get("players") or []
    return ball, players


def _marks_from_clips(video):
    root = os.path.join(os.path.dirname(os.path.abspath(video)), "events", _stem(video))
    marks = {}
    if not os.path.isdir(root):
        return marks
    for clip in sorted(glob.glob(os.path.join(root, "clip_*"))):
        pj = os.path.join(clip, "players.json")
        lj = os.path.join(clip, "Label.csv")
        if not (os.path.isfile(pj) and os.path.isfile(lj)):
            continue
        with open(pj, encoding="utf-8") as f:
            frames = json.load(f).get("frames") or []
        with open(lj, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for fr, row in zip(frames, rows):
            st = int(row["status"])
            if st in (1, 2):
                marks[int(fr["src_frame"])] = st
    return marks


def _sync_clips(video, marks):
    root = os.path.join(os.path.dirname(os.path.abspath(video)), "events", _stem(video))
    if not os.path.isdir(root):
        return
    for clip in sorted(glob.glob(os.path.join(root, "clip_*"))):
        pj = os.path.join(clip, "players.json")
        lj = os.path.join(clip, "Label.csv")
        if not (os.path.isfile(pj) and os.path.isfile(lj)):
            continue
        with open(pj, encoding="utf-8") as f:
            frames = json.load(f).get("frames") or []
        with open(lj, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != len(frames):
            continue
        for row, fr in zip(rows, frames):
            row["status"] = str(int(marks.get(int(fr["src_frame"]), 0)))
        with open(lj, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file name", "visibility", "x-coordinate", "y-coordinate", "status"])
            w.writeheader()
            w.writerows(rows)


def _save(video, marks):
    path = _marks_path(video)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "status"])
        for fid in sorted(marks):
            if marks[fid] in (1, 2):
                w.writerow([fid, marks[fid]])
    _sync_clips(video, {k: v for k, v in marks.items() if v in (1, 2)})
    print("saved", path, "events", sum(1 for v in marks.values() if v in (1, 2)))


def _read(cap, fid, pos):
    if pos != fid:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ok, frame = cap.read()
    return ok, frame, fid + 1


def _draw(frame, fid, marks, ball, players, scale):
    view = frame
    if scale != 1.0:
        view = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
    for d in range(-12, 13):
        pt = ball.get(fid + d)
        if pt is None:
            continue
        x, y = int(pt[0] * scale), int(pt[1] * scale)
        color = (0, 255, 0) if d == 0 else ((255, 180, 0) if d < 0 else (0, 180, 255))
        r = 8 if d == 0 else 3
        cv2.circle(view, (x, y), r, color, -1)
    for p in players.get(fid) or []:
        b = p.get("bbox") or []
        if len(b) < 4:
            continue
        cv2.rectangle(
            view,
            (int(b[0] * scale), int(b[1] * scale)),
            (int(b[2] * scale), int(b[3] * scale)),
            (255, 0, 180) if p.get("side") == "far" else (0, 165, 255),
            2,
        )
    st = marks.get(fid, 0)
    name = {0: "flight", 1: "HIT", 2: "BOUNCE"}.get(st, "flight")
    color = {0: (0, 255, 0), 1: (0, 0, 255), 2: (0, 255, 255)}[st if st in (0, 1, 2) else 0]
    cv2.rectangle(view, (0, 0), (view.shape[1], 72), (0, 0, 0), -1)
    cv2.putText(view, f"frame {fid}  {name}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(
        view,
        "space play   a/d step   q/e +/-10   1 hit  2 bounce  0 clear   n/p jump   esc save",
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    return view


def _jump(keys, fid, step):
    if not keys:
        return fid
    if step > 0:
        nxt = [k for k in keys if k > fid]
        return nxt[0] if nxt else keys[-1]
    prv = [k for k in keys if k < fid]
    return prv[-1] if prv else keys[0]


def run(video):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    marks = _load_marks(_marks_path(video)) or _marks_from_clips(video)
    ball, players = _load_tracks(video)
    fid = 0
    pos = None
    paused = True
    scale = 1.0
    win = "label " + _stem(video)
    while True:
        ok, frame, pos = _read(cap, fid, pos)
        if not ok:
            break
        h, w = frame.shape[:2]
        scale = min(1.0, 1400.0 / max(w, 1))
        cv2.imshow(win, _draw(frame, fid, marks, ball, players, scale))
        key = cv2.waitKeyEx(0 if paused else max(1, int(1000 / fps)))
        if key in (-1, 0):
            if not paused:
                fid = min(total - 1, fid + 1)
                if fid >= total - 1:
                    paused = True
            continue
        code = key & 0xFF
        if key == 27 or code == 27:
            break
        if code == ord(" "):
            paused = not paused
        elif code in (ord("a"),) or key == 2424832:
            paused = True
            fid = max(0, fid - 1)
        elif code in (ord("d"),) or key == 2555904:
            paused = True
            fid = min(total - 1, fid + 1)
        elif code == ord("q"):
            paused = True
            fid = max(0, fid - 10)
        elif code == ord("e"):
            paused = True
            fid = min(total - 1, fid + 10)
        elif code == ord("1"):
            marks[fid] = 1
            paused = True
            _save(video, marks)
        elif code == ord("2"):
            marks[fid] = 2
            paused = True
            _save(video, marks)
        elif code == ord("0"):
            marks.pop(fid, None)
            paused = True
            _save(video, marks)
        elif code in (ord("n"), ord("p")):
            paused = True
            keys = sorted(k for k, v in marks.items() if v in (1, 2))
            if keys:
                fid = _jump(keys, fid, 1 if code == ord("n") else -1)
            else:
                fid = min(total - 1, fid + 15) if code == ord("n") else max(0, fid - 15)
    _save(video, marks)
    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args().video)
