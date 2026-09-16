from typing import List, Optional

from .bounce import detect_bounces
from .court3d import DB, NET, classify_xy, estimate_camera, ray_at_z
from .geom import ball_center


def _near_racket(uv, players_frame, pad: float = 35.0) -> bool:
    """球在球员上半身/拍附近才当击球；脚边落地不要滤掉。"""
    if uv is None or not players_frame:
        return False
    x, y = float(uv[0]), float(uv[1])
    for p in players_frame:
        b = p.get("bbox") or []
        if len(b) < 4:
            continue
        x1, y1, x2, y2 = b
        h = max(1.0, y2 - y1)
        # 上 55%：躯干+拍；下 45%：腿脚，真落地常在这里
        y_cut = y1 + 0.55 * h
        if x1 - pad <= x <= x2 + pad and y1 - pad <= y <= y_cut + pad:
            return True
    return False


def _snap_landing(filled, fid: int, w: int = 5) -> int:
    """落到图像 y 局部最大（高位机位：球最低点）附近。"""
    n = len(filled)
    best, best_y = fid, -1.0
    for j in range(max(0, fid - w), min(n, fid + w + 1)):
        uv = ball_center(filled[j].get(1, []))
        if uv is None:
            continue
        if uv[1] > best_y:
            best_y = uv[1]
            best = j
    return best


def analyze_score(
    filled,
    court_dets,
    fps: float,
    bounce_weights: Optional[str] = None,
    image_shape=None,
    players=None,
    shot_frames: Optional[List[int]] = None,
):
    bounce_ids = detect_bounces(filled, bounce_weights)
    bounces = []
    cam = None
    last = -999
    # tennis-vision: ~400ms 去重；略放宽以免合并两次真落地
    min_gap = max(10, int(round(0.33 * float(fps or 30))))
    seen = set()
    for raw_fid in bounce_ids:
        if raw_fid < 0 or raw_fid >= len(filled):
            continue
        fid = _snap_landing(filled, int(raw_fid))
        if fid in seen:
            continue
        if fid - last < min_gap:
            continue
        uv = ball_center(filled[fid].get(1, []))
        if uv is None:
            continue
        pl = players[fid] if players and fid < len(players) else None
        if _near_racket(uv, pl):
            continue
        det = court_dets[fid] if court_dets and fid < len(court_dets) else None
        if det is not None and getattr(det, "quality_ok", False) and image_shape is not None:
            got = estimate_camera(det.keypoints_xy, image_shape)
            if got is not None:
                cam = got
        xyz = ray_at_z(cam, uv, 0.0)
        if xyz is None:
            continue
        # 只丢掉明显飞到看台/场外很远的噪声
        if abs(xyz[0]) > DB + 2.5 or abs(xyz[1]) > NET + 3.0:
            continue
        info = classify_xy(xyz[0], xyz[1], eps=0.12)
        seen.add(fid)
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
