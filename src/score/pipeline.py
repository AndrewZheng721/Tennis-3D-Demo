from typing import List, Optional

from .bounce import detect_bounces
from .geom import H_inv_of, ball_center, classify_bounce, image_to_court
from .rally import run_rally


def analyze_score(filled, court_dets, fps: float, bounce_weights: Optional[str] = None):
    bounce_ids = detect_bounces(filled, bounce_weights)
    bounces = []
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
        bounces.append(rec)
    events = run_rally(bounces)
    return {
        "fps": float(fps),
        "num_bounces": len(bounces),
        "bounces": bounces,
        "events": events,
        "points": [e for e in events if e["type"] == "point"],
    }
