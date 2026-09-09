from typing import List, Optional

import cv2
import numpy as np

from .court3d import COURT_LINES_3D, COURT_XYZ, DB, NET


def _to_px(X, Y, Z, w, h):
    px = w * 0.5 + X * (w / 14.5)
    py = h * 0.84 - (Y + NET) * (h / 30.5) - Z * (h / 9.5)
    return int(round(px)), int(round(py))


def _polyline(img, pts, color, thickness=2):
    if len(pts) < 2:
        return
    cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)


def draw_court3d(w: int, h: int, xyz, trail: List[Optional[tuple]], bounce_set=None):
    vis = np.full((h, w, 3), 28, dtype=np.uint8)

    def p(i, z=0.0):
        return _to_px(COURT_XYZ[i, 0], COURT_XYZ[i, 1], z, w, h)

    fill = np.array(
        [p(0), p(1), p(3), p(2)], dtype=np.int32
    )
    cv2.fillConvexPoly(vis, fill, (62, 92, 36))
    inner = np.array([p(4), p(6), p(7), p(5)], dtype=np.int32)
    cv2.fillConvexPoly(vis, inner, (168, 92, 42))
    for a, b in COURT_LINES_3D:
        cv2.line(vis, p(a), p(b), (230, 230, 230), 1, cv2.LINE_AA)
    net = [p(0, 0), _to_px(-DB, 0, 1.07, w, h), _to_px(DB, 0, 1.07, w, h), p(1, 0)]
    _polyline(vis, net, (200, 200, 200), 2)
    cv2.line(vis, _to_px(-DB, 0, 0, w, h), _to_px(DB, 0, 0, w, h), (180, 180, 180), 1)

    shadow = []
    air = []
    for q in trail:
        if q is None:
            if len(shadow) >= 2:
                _polyline(vis, shadow, (40, 40, 40), 1)
            if len(air) >= 2:
                _polyline(vis, air, (0, 220, 255), 2)
            shadow, air = [], []
            continue
        shadow.append(_to_px(q[0], q[1], 0.0, w, h))
        air.append(_to_px(q[0], q[1], q[2], w, h))
    if len(shadow) >= 2:
        _polyline(vis, shadow, (40, 40, 40), 1)
    if len(air) >= 2:
        _polyline(vis, air, (0, 220, 255), 2)

    if xyz is not None:
        sx, sy = _to_px(xyz[0], xyz[1], 0.0, w, h)
        bx, by = _to_px(xyz[0], xyz[1], xyz[2], w, h)
        cv2.line(vis, (sx, sy), (bx, by), (90, 90, 90), 1, cv2.LINE_AA)
        cv2.circle(vis, (sx, sy), 4, (70, 70, 70), -1)
        cv2.circle(vis, (bx, by), 6, (0, 255, 255), -1)
        cv2.putText(
            vis,
            f"X {xyz[0]:.2f}  Y {xyz[1]:.2f}  Z {xyz[2]:.2f} m",
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
    else:
        cv2.putText(vis, "no 3D", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(vis, "3D court (m)", (24, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    return vis
