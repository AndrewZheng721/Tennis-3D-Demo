import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _fix_path(video):
    return os.path.join(os.path.dirname(os.path.abspath(video)), "labels_ball", _stem(video) + ".csv")


def _load_fixes(path):
    fixes = {}
    if not os.path.isfile(path):
        return fixes
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fixes[int(row["frame"])] = (int(float(row["x"])), int(float(row["y"])), int(row["vis"]))
    return fixes


def _load_auto(video):
    root = os.path.join(os.path.dirname(os.path.abspath(video)), "events", _stem(video))
    ball = {}
    if not os.path.isdir(root):
        return ball
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
            if int(row["visibility"]) > 0:
                ball[int(fr["src_frame"])] = (int(row["x-coordinate"]), int(row["y-coordinate"]))
    return ball


def _at(fid, ball, fixes):
    if fid in fixes:
        x, y, vis = fixes[fid]
        if vis <= 0:
            return None
        return (x, y)
    return ball.get(fid)


def _jumps(ball, fixes, total, thresh=100):
    out = []
    prev = None
    prev_f = None
    for fid in range(total):
        pt = _at(fid, ball, fixes)
        if pt is None:
            continue
        if prev is not None and fid - prev_f <= 2:
            if float(np.hypot(pt[0] - prev[0], pt[1] - prev[1])) > thresh:
                out.append(fid)
        prev, prev_f = pt, fid
    return out


def _save(video, fixes):
    path = _fix_path(video)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x", "y", "vis"])
        for fid in sorted(fixes):
            x, y, vis = fixes[fid]
            w.writerow([fid, x, y, vis])
    print("saved", path, "fixes", len(fixes))


def _read(cap, fid, pos):
    if pos != fid:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ok, frame = cap.read()
    return ok, frame, fid + 1


def _draw(frame, fid, ball, fixes, jumps, scale):
    view = frame
    if scale != 1.0:
        view = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
    pts = []
    for d in range(-15, 16):
        pt = _at(fid + d, ball, fixes)
        pts.append(None if pt is None else (int(pt[0] * scale), int(pt[1] * scale)))
    cur = pts[15]
    overlay = view.copy()
    seq = []
    for p in pts:
        near = False
        if p is not None and cur is not None:
            near = (p[0] - cur[0]) ** 2 + (p[1] - cur[1]) ** 2 <= 16 * 16
        if p is None or near:
            if len(seq) >= 2:
                cv2.polylines(overlay, [np.array(seq, np.int32)], False, (255, 160, 0), 1, cv2.LINE_AA)
            seq = []
        else:
            seq.append(p)
    if len(seq) >= 2:
        cv2.polylines(overlay, [np.array(seq, np.int32)], False, (255, 160, 0), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, view, 0.65, 0, view)
    if cur is not None:
        color = (0, 255, 255) if fid in fixes else (0, 255, 0)
        cv2.circle(view, cur, 12, color, 1, cv2.LINE_AA)
    if fid in jumps:
        cv2.circle(view, (24, 24), 10, (0, 0, 255), -1)
    gone = fid in fixes and fixes[fid][2] <= 0
    tag = "NO BALL" if gone else ("fixed" if fid in fixes else "auto")
    h, w = view.shape[:2]
    cv2.rectangle(view, (0, h - 64), (w, h), (0, 0, 0), -1)
    cv2.putText(view, f"frame {fid}  {tag}", (16, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(
        view,
        "click ball   0 no ball   u undo   n/p jump   space play   a/d step   q/e +/-10   esc save",
        (16, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    return view


def _jump_to(keys, fid, step):
    if not keys:
        return fid
    if step > 0:
        nxt = [k for k in keys if k > fid]
        return nxt[0] if nxt else keys[-1]
    prv = [k for k in keys if k < fid]
    return prv[-1] if prv else keys[0]


def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param["click"] = (x, y)


def _click_xy(win, click, view_w, view_h, scale, frame_w, frame_h):
    raw = (float(click[0]), float(click[1]))
    cands = [raw]
    try:
        rx, ry, rw, rh = cv2.getWindowImageRect(win)
        if rw > 1 and rh > 1:
            cands.append(((raw[0] - rx) * view_w / float(rw), (raw[1] - ry) * view_h / float(rh)))
    except Exception:
        pass
    picked = None
    for x, y in cands:
        if 0 <= x < view_w and 0 <= y < view_h:
            picked = (x, y)
            break
    if picked is None:
        x, y = cands[-1]
        picked = (min(max(x, 0), view_w - 1), min(max(y, 0), view_h - 1))
    ox = min(max(int(round(picked[0] / scale)), 0), frame_w - 1)
    oy = min(max(int(round(picked[1] / scale)), 0), frame_h - 1)
    return ox, oy


def run(video):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ball = _load_auto(video)
    fixes = _load_fixes(_fix_path(video))
    jumps = _jumps(ball, fixes, total)
    fid = jumps[0] if jumps else 0
    pos = None
    paused = True
    scale = 1.0
    mouse = {"click": None, "cb": _on_mouse}
    shown = None
    shown_fid = None
    hooked = False
    win = "ball " + _stem(video)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        if shown_fid != fid:
            ok, frame, pos = _read(cap, fid, pos)
            if not ok:
                break
            shown = frame
            shown_fid = fid
        frame = shown
        h, w = frame.shape[:2]
        scale = min(1.0, 1400.0 / max(w, 1))
        view = _draw(frame, fid, ball, fixes, set(jumps), scale)
        cv2.imshow(win, view)
        if not hooked:
            cv2.waitKey(1)
            cv2.setMouseCallback(win, mouse["cb"], mouse)
            hooked = True
        key = cv2.waitKeyEx(30 if paused else max(1, int(1000 / fps)))
        click = mouse["click"]
        mouse["click"] = None
        if click is not None:
            pt = _click_xy(win, click, view.shape[1], view.shape[0], scale, w, h)
            fixes[fid] = (pt[0], pt[1], 1)
            paused = True
            print("click", fid, pt[0], pt[1])
            jumps = _jumps(ball, fixes, total)
            _save(video, fixes)
            continue
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
        elif code == ord("0"):
            fixes[fid] = (0, 0, 0)
            paused = True
            jumps = _jumps(ball, fixes, total)
            _save(video, fixes)
        elif code == ord("u"):
            fixes.pop(fid, None)
            paused = True
            jumps = _jumps(ball, fixes, total)
            _save(video, fixes)
        elif code in (ord("n"), ord("p")):
            paused = True
            fid = _jump_to(jumps, fid, 1 if code == ord("n") else -1)
    _save(video, fixes)
    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args().video)
