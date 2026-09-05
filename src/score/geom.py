from typing import Optional, Tuple

import cv2
import numpy as np

REF_KPS = np.array(
    [
        [286, 561],
        [1379, 561],
        [286, 2935],
        [1379, 2935],
        [423, 561],
        [423, 2935],
        [1242, 561],
        [1242, 2935],
        [423, 1110],
        [1242, 1110],
        [423, 2386],
        [1242, 2386],
        [832, 1110],
        [832, 2386],
    ],
    dtype=np.float32,
)

NET_Y = float((REF_KPS[12, 1] + REF_KPS[13, 1]) / 2.0)
NET_X = float(REF_KPS[12, 0])

SINGLES = REF_KPS[[4, 6, 7, 5]].astype(np.float32)
DOUBLES = REF_KPS[[0, 1, 3, 2]].astype(np.float32)

_NL = np.array([REF_KPS[4, 0], NET_Y], dtype=np.float32)
_NR = np.array([REF_KPS[6, 0], NET_Y], dtype=np.float32)
_NC = np.array([NET_X, NET_Y], dtype=np.float32)

SERVICE = {
    "far_left": np.stack([REF_KPS[8], REF_KPS[12], _NC, _NL]),
    "far_right": np.stack([REF_KPS[12], REF_KPS[9], _NR, _NC]),
    "near_left": np.stack([_NL, _NC, REF_KPS[13], REF_KPS[10]]),
    "near_right": np.stack([_NC, _NR, REF_KPS[11], REF_KPS[13]]),
}


def image_to_court(xy, H_inv) -> Optional[Tuple[float, float]]:
    if H_inv is None or xy is None:
        return None
    p = np.array([[[float(xy[0]), float(xy[1])]]], dtype=np.float32)
    q = cv2.perspectiveTransform(p, H_inv.astype(np.float32))[0, 0]
    if not np.isfinite(q).all():
        return None
    return float(q[0]), float(q[1])


def _inside(xy, poly) -> bool:
    if xy is None:
        return False
    return cv2.pointPolygonTest(poly.astype(np.float32), (float(xy[0]), float(xy[1])), False) >= 0


def classify_bounce(court_xy) -> dict:
    if court_xy is None:
        return {
            "side": "unknown",
            "in_singles": False,
            "in_doubles": False,
            "service": None,
            "call": "unknown",
        }
    x, y = court_xy
    side = "far" if y < NET_Y else "near"
    in_s = _inside(court_xy, SINGLES)
    in_d = _inside(court_xy, DOUBLES)
    svc = None
    for name, poly in SERVICE.items():
        if _inside(court_xy, poly):
            svc = name
            break
    call = "in" if in_s else "out"
    return {
        "side": side,
        "in_singles": in_s,
        "in_doubles": in_d,
        "service": svc,
        "call": call,
    }


def ball_center(box) -> Optional[Tuple[float, float]]:
    if not box or len(box) < 4 or not np.isfinite(box[0]):
        return None
    return float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0)


def H_inv_of(det) -> Optional[np.ndarray]:
    if det is None:
        return None
    return getattr(det, "homography_image_to_ref", None)
