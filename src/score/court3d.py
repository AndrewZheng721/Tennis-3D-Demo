from typing import Optional, Tuple

import cv2
import numpy as np

HW = 10.97
HL = 23.77
SW = 8.23
SL = 6.40
NET = 11.885
SING = 4.115
DB = 5.485

COURT_XYZ = np.array(
    [
        [-DB, NET, 0],
        [DB, NET, 0],
        [-DB, -NET, 0],
        [DB, -NET, 0],
        [-SING, NET, 0],
        [-SING, -NET, 0],
        [SING, NET, 0],
        [SING, -NET, 0],
        [-SING, SL, 0],
        [SING, SL, 0],
        [-SING, -SL, 0],
        [SING, -SL, 0],
        [0.0, SL, 0],
        [0.0, -SL, 0],
    ],
    dtype=np.float64,
)

COURT_LINES_3D = [
    (0, 1),
    (2, 3),
    (0, 2),
    (1, 3),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
    (4, 6),
    (5, 7),
    (12, 13),
    (8, 10),
    (9, 11),
]


def estimate_camera(keypoints_xy, image_shape) -> Optional[dict]:
    pts = np.asarray(keypoints_xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 4 or not np.isfinite(pts).all():
        return None
    h, w = image_shape[:2]
    cx, cy = w * 0.5, h * 0.5
    obj = COURT_XYZ[: pts.shape[0]]
    img = pts[: obj.shape[0]].astype(np.float64)
    n = min(len(obj), len(img))
    obj, img = obj[:n], img[:n]
    best = None
    for f in np.linspace(700.0, 3800.0, 28):
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]], dtype=np.float64)
        ok, rvec, tvec = False, None, None
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP
            )
        except cv2.error:
            pass
        if not ok:
            ok, rvec, tvec = cv2.solvePnP(
                obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE
            )
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)))
        if best is None or err < best[0]:
            best = (err, f, rvec, tvec, K)
    if best is None or best[0] > 25.0:
        return None
    err, f, rvec, tvec, K = best
    R, _ = cv2.Rodrigues(rvec)
    C = (-R.T @ tvec.reshape(3)).reshape(3)
    if C[2] < 2.0 or C[2] > 40.0:
        return None
    return {
        "K": K,
        "R": R,
        "t": tvec.reshape(3),
        "rvec": rvec,
        "C": C,
        "f": float(f),
        "err": err,
    }


def ray_at_z(cam: dict, uv, z: float) -> Optional[Tuple[float, float, float]]:
    if cam is None or uv is None:
        return None
    u, v = float(uv[0]), float(uv[1])
    K = cam["K"]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    d_w = cam["R"].T @ d_cam
    C = cam["C"]
    if abs(d_w[2]) < 1e-8:
        return None
    lam = (z - C[2]) / d_w[2]
    if lam <= 0:
        return None
    P = C + lam * d_w
    if not np.isfinite(P).all():
        return None
    if abs(P[0]) > 20 or abs(P[1]) > 25 or P[2] < -1 or P[2] > 20:
        return None
    return float(P[0]), float(P[1]), float(P[2])


def classify_xy(x: float, y: float, eps: float = 0.12) -> dict:
    in_singles = abs(x) <= SING + eps and abs(y) <= NET + eps
    in_doubles = abs(x) <= DB + eps and abs(y) <= NET + eps
    serve = None
    if -SING - eps <= x <= 0.0 + eps and 0.0 - eps <= y <= SL + eps:
        serve = "far_left"
    elif 0.0 - eps <= x <= SING + eps and 0.0 - eps <= y <= SL + eps:
        serve = "far_right"
    elif -SING - eps <= x <= 0.0 + eps and -SL - eps <= y <= 0.0 + eps:
        serve = "near_left"
    elif 0.0 - eps <= x <= SING + eps and -SL - eps <= y <= 0.0 + eps:
        serve = "near_right"
    if not in_singles:
        serve = None
        rally = "out"
        side = "far" if y >= 0 else "near"
    else:
        rally = "far_in" if y >= 0 else "near_in"
        side = "far" if y >= 0 else "near"
        if serve and ((serve.startswith("far") and y < 0) or (serve.startswith("near") and y > 0)):
            serve = None
    return {
        "side": side,
        "serve_zone": serve,
        "rally_zone": rally,
        "in_singles": in_singles,
        "in_doubles": in_doubles,
    }
