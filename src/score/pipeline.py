from typing import List, Optional

from .bounce import detect_bounces
from .court3d import DB, NET, classify_xy, estimate_camera, ray_at_z
from .geom import ball_center


def _near_player(uv, players_frame, pad: float = 70.0) -> bool:
    if uv is None or not players_frame:
        return False
    x, y = float(uv[0]), float(uv[1])
    for p in players_frame:
        b = p.get("bbox") or []
        if len(b) < 4:
            continue
        if b[0] - pad <= x <= b[2] + pad and b[1] - pad <= y <= b[3] + pad:
            return True
    return False


def _is_hit_turn(filled, fid: int, w: int = 4) -> bool:
    """击球附近球心常靠近人，且 2D 速度在短窗内反向；用轨迹折点幅度辅助。"""
    n = len(filled)
    if fid < w or fid + w >= n:
        return False
    pts = []
    for j in range(fid - w, fid + w + 1):
        uv = ball_center(filled[j].get(1, []))
        if uv is None:
            return False
        pts.append(uv)
    mid = w
    v0 = (pts[mid][0] - pts[0][0], pts[mid][1] - pts[0][1])
    v1 = (pts[-1][0] - pts[mid][0], pts[-1][1] - pts[mid][1])
    n0 = (v0[0] ** 2 + v0[1] ** 2) ** 0.5
    n1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
    if n0 < 25 or n1 < 25:
        return False
    dot = v0[0] * v1[0] + v0[1] * v1[1]
    return dot < 0 and n0 > 40 and n1 > 40


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
    shot_set = set(shot_frames or [])
    bounces = []
    cam = None
    last = -999
    min_gap = max(12, int(round(0.4 * float(fps or 30))))
    for fid in bounce_ids:
        if fid < 0 or fid >= len(filled):
            continue
        if fid - last < min_gap:
            continue
        uv = ball_center(filled[fid].get(1, []))
        if uv is None:
            continue
        # 击球帧邻域 / 球靠近球员框 → 不是落地
        if any(abs(fid - s) <= 8 for s in shot_set):
            continue
        pl = players[fid] if players and fid < len(players) else None
        if _near_player(uv, pl):
            continue
        if _is_hit_turn(filled, fid):
            # 折点且离人近才丢；纯折点仍可能是真落地
            if pl and _near_player(uv, pl, pad=120.0):
                continue
        det = court_dets[fid] if court_dets and fid < len(court_dets) else None
        if det is not None and getattr(det, "quality_ok", False) and image_shape is not None:
            got = estimate_camera(det.keypoints_xy, image_shape)
            if got is not None:
                cam = got
        xyz = ray_at_z(cam, uv, 0.0)
        if xyz is None:
            continue
        # 场外过远：检测噪声（tennis-vision / CourtCheck 同类门控）
        if abs(xyz[0]) > DB + 1.5 or abs(xyz[1]) > NET + 2.0:
            continue
        if abs(xyz[0]) > 8.0 or abs(xyz[1]) > 14.0:
            continue
        info = classify_xy(xyz[0], xyz[1], eps=0.12)
        # 明显飞出双打区很远的「out」更像误检，丢掉
        if not info["in_doubles"] and (abs(xyz[0]) > DB + 0.8 or abs(xyz[1]) > NET + 1.0):
            continue
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
