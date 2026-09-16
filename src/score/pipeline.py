from typing import List, Optional

from .bounce import _centers, _geom_bounces, detect_bounces
from .court3d import DB, NET, classify_xy, estimate_camera, ray_at_z
from .geom import ball_center


def _near_racket(uv, players_frame) -> bool:
    if uv is None or not players_frame:
        return False
    x, y = float(uv[0]), float(uv[1])
    for p in players_frame:
        b = p.get("bbox") or []
        if len(b) < 4:
            continue
        x1, y1, x2, y2 = b
        h = max(1.0, y2 - y1)
        pad = max(28.0, 0.22 * h)
        y_cut = y1 + 0.60 * h
        if x1 - pad <= x <= x2 + pad and y1 - pad <= y <= y_cut + pad:
            return True
    return False


def _ball_side(xyz, uv=None, image_shape=None) -> Optional[str]:
    if xyz is not None:
        return "far" if xyz[1] >= 0 else "near"
    if uv is not None and image_shape is not None:
        return "far" if uv[1] < image_shape[0] * 0.42 else "near"
    return None


def _snap_landing(filled, fid: int, cam=None, image_shape=None, w: int = 5) -> int:
    n = len(filled)
    uv0 = ball_center(filled[fid].get(1, []))
    xyz0 = ray_at_z(cam, uv0, 0.0) if cam is not None else None
    far = _ball_side(xyz0, uv0, image_shape) == "far"
    best = fid
    best_y = None
    for j in range(max(0, fid - w), min(n, fid + w + 1)):
        uv = ball_center(filled[j].get(1, []))
        if uv is None:
            continue
        y = uv[1]
        if best_y is None:
            best, best_y = j, y
        elif far and y < best_y:
            best, best_y = j, y
        elif (not far) and y > best_y:
            best, best_y = j, y
    return best


def _update_cam(cam, det, image_shape):
    if det is None or image_shape is None or not getattr(det, "quality_ok", False):
        return cam
    got = estimate_camera(det.keypoints_xy, image_shape)
    return got if got is not None else cam


def _xyz_y(filled, fid, cam):
    uv = ball_center(filled[fid].get(1, []))
    xyz = ray_at_z(cam, uv, 0.0) if cam is not None else None
    return uv, xyz


def _court_y_reversed(filled, fid, cam, span: int = 3) -> bool:
    if cam is None:
        return False
    n = len(filled)
    ys = []
    for j in range(max(0, fid - span), min(n, fid + span + 1)):
        _, xyz = _xyz_y(filled, j, cam)
        ys.append(None if xyz is None else xyz[1])
    mid = fid - max(0, fid - span)
    y0 = ys[mid] if 0 <= mid < len(ys) else None
    if y0 is None:
        return False
    pre = [v for v in ys[:mid] if v is not None]
    post = [v for v in ys[mid + 1 :] if v is not None]
    if len(pre) < 2 or len(post) < 2:
        return False
    vy_in = y0 - pre[0]
    vy_out = post[-1] - y0
    if vy_in * vy_out >= 0:
        return False
    return min(abs(vy_in), abs(vy_out)) >= 0.6


def analyze_score(
    filled,
    court_dets,
    fps: float,
    bounce_weights: Optional[str] = None,
    image_shape=None,
    players=None,
    shot_frames: Optional[List[int]] = None,
):
    bounce_ids = set(detect_bounces(filled, bounce_weights))
    xs, ys = _centers(filled)
    hit_ids = set(_geom_bounces(xs, ys, min_gap=6))
    bounces = []
    cam = None
    last = -999
    last_hit_side = None
    last_bounce_side = None
    min_gap = max(10, int(round(0.33 * float(fps or 30))))
    seen = set()
    n = len(filled)
    for raw_fid in sorted(bounce_ids | hit_ids):
        if raw_fid < 0 or raw_fid >= n:
            continue
        det = court_dets[raw_fid] if court_dets and raw_fid < len(court_dets) else None
        cam = _update_cam(cam, det, image_shape)
        fid = _snap_landing(filled, int(raw_fid), cam, image_shape)
        if fid in seen:
            continue
        det = court_dets[fid] if court_dets and fid < len(court_dets) else None
        cam = _update_cam(cam, det, image_shape)
        uv, xyz = _xyz_y(filled, fid, cam)
        if uv is None:
            continue
        pl = players[fid] if players and fid < len(players) else None
        side = _ball_side(xyz, uv, image_shape)
        is_hit = _near_racket(uv, pl) or _court_y_reversed(filled, fid, cam)
        if is_hit:
            if side:
                last_hit_side = side
            continue
        if raw_fid not in bounce_ids:
            continue
        if fid - last < min_gap:
            continue
        if xyz is None:
            continue
        if abs(xyz[0]) > DB + 2.5 or abs(xyz[1]) > NET + 3.0:
            continue
        if last_hit_side and side == last_hit_side:
            continue
        if last_bounce_side and side == last_bounce_side:
            continue
        info = classify_xy(xyz[0], xyz[1], eps=0.12)
        seen.add(fid)
        last = fid
        last_bounce_side = side
        last_hit_side = None
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
