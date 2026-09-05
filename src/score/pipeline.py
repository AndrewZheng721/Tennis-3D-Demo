from typing import List, Optional

from .bounce import detect_bounces
from .geom import NET_Y, H_inv_of, ball_center, classify_bounce, image_to_court
from .rally import run_rally


def _court_y_at(filled, court_dets, fid):
    if fid < 0 or fid >= len(filled):
        return None
    xy = ball_center(filled[fid].get(1, []))
    det = court_dets[fid] if court_dets and fid < len(court_dets) else None
    cxy = image_to_court(xy, H_inv_of(det))
    return None if cxy is None else cxy[1]


def _is_hit(filled, court_dets, fid, side, lookahead=30):
    y0 = _court_y_at(filled, court_dets, fid)
    toward = 0
    for j in range(fid + 1, min(len(filled), fid + lookahead + 1)):
        y = _court_y_at(filled, court_dets, j)
        if y is None or y0 is None:
            continue
        if side == "far" and (y > y0 + 35 or y > NET_Y):
            toward += 1
        elif side == "near" and (y < y0 - 35 or y < NET_Y):
            toward += 1
    return toward >= 3


def analyze_score(filled, court_dets, fps: float, bounce_weights: Optional[str] = None):
    bounce_ids = detect_bounces(filled, bounce_weights)
    bounces = []
    hits = []
    for fid in bounce_ids:
        if fid < 0 or fid >= len(filled):
            continue
        xy = ball_center(filled[fid].get(1, []))
        det = court_dets[fid] if court_dets and fid < len(court_dets) else None
        court_xy = image_to_court(xy, H_inv_of(det))
        info = classify_bounce(court_xy)
        rec = {
            "frame_id": int(fid),
            "time_sec": round(fid / fps, 4) if fps else fid,
            "image_xy": None if xy is None else [xy[0], xy[1]],
            "court_xy": None if court_xy is None else [court_xy[0], court_xy[1]],
            **info,
        }
        if rec["side"] in ("near", "far") and _is_hit(filled, court_dets, fid, rec["side"]):
            rec["kind"] = "hit"
            hits.append(rec)
            continue
        rec["kind"] = "bounce"
        bounces.append(rec)
    events = run_rally(bounces)
    return {
        "fps": float(fps),
        "num_bounces": len(bounces),
        "bounces": bounces,
        "hits": hits,
        "events": events,
        "points": [e for e in events if e["type"] == "point"],
    }
