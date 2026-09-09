from typing import Optional

from .bounce import detect_bounces
from .court3d import classify_xy, estimate_camera, ray_at_z
from .geom import ball_center


def analyze_score(filled, court_dets, fps: float, bounce_weights: Optional[str] = None, image_shape=None):
    bounce_ids = detect_bounces(filled, bounce_weights)
    bounces = []
    cam = None
    last = -999
    for fid in bounce_ids:
        if fid < 0 or fid >= len(filled):
            continue
        if fid - last < 12:
            continue
        uv = ball_center(filled[fid].get(1, []))
        if uv is None:
            continue
        det = court_dets[fid] if court_dets and fid < len(court_dets) else None
        if det is not None and getattr(det, "quality_ok", False) and image_shape is not None:
            got = estimate_camera(det.keypoints_xy, image_shape)
            if got is not None:
                cam = got
        xyz = ray_at_z(cam, uv, 0.0)
        if xyz is None:
            continue
        if abs(xyz[0]) > 8.0 or abs(xyz[1]) > 14.0:
            continue
        info = classify_xy(xyz[0], xyz[1])
        last = fid
        bounces.append(
            {
                "frame_id": int(fid),
                "time_sec": round(fid / fps, 4) if fps else fid,
                "image_xy": [uv[0], uv[1]],
                "xyz": [xyz[0], xyz[1], xyz[2]],
                **info,
            }
        )
    return {
        "fps": float(fps),
        "num_bounces": len(bounces),
        "bounces": bounces,
    }
