from typing import Optional

import numpy as np

from .bounce import detect_bounces
from .court3d import estimate_camera, ray_at_z
from .geom import ball_center

G = 9.81


def _cam_for(det, image_shape, prev):
    if det is None or not getattr(det, "quality_ok", False):
        return prev
    cam = estimate_camera(det.keypoints_xy, image_shape)
    return cam if cam is not None else prev


def reconstruct_ball_3d(filled, court_dets, fps: float, image_shape, bounce_weights=None, bounce_ids=None):
    n = len(filled)
    if bounce_ids is None:
        bounce_ids = detect_bounces(filled, bounce_weights)
    bounce_ids = sorted(i for i in bounce_ids if 0 <= i < n)
    cams = []
    prev = None
    for i in range(n):
        det = court_dets[i] if court_dets and i < len(court_dets) else None
        prev = _cam_for(det, image_shape, prev)
        cams.append(prev)

    z_of = np.full(n, np.nan, dtype=np.float64)
    for a, b in zip(bounce_ids, bounce_ids[1:]):
        dt = (b - a) / float(fps)
        if dt < 0.12 or dt > 4.0:
            continue
        vz = 0.5 * G * dt
        for t in range(a, b + 1):
            tau = (t - a) / float(fps)
            z_of[t] = max(0.0, vz * tau - 0.5 * G * tau * tau)

    frames = []
    for i in range(n):
        uv = ball_center(filled[i].get(1, []))
        cam = cams[i]
        gxyz = ray_at_z(cam, uv, 0.0) if uv is not None else None
        z = z_of[i]
        if uv is not None and cam is not None and np.isfinite(z):
            xyz = ray_at_z(cam, uv, float(z))
        else:
            xyz = gxyz
        frames.append(
            {
                "frame_id": i,
                "time_sec": round(i / fps, 4) if fps else i,
                "xyz": None if xyz is None else [xyz[0], xyz[1], xyz[2]],
                "xyz_ground": None if gxyz is None else [gxyz[0], gxyz[1], gxyz[2]],
                "z_physics": None if not np.isfinite(z) else float(z),
            }
        )
    return {
        "fps": float(fps),
        "unit": "meter",
        "origin": "court_center_net_ground",
        "axes": {"x": "right", "y": "far", "z": "up"},
        "bounce_frames": bounce_ids,
        "frames": frames,
    }
