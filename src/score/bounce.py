import os
from typing import List, Optional, Tuple

import numpy as np


def _centers(filled) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    xs, ys = [], []
    for item in filled:
        box = item.get(1, [])
        if box and len(box) >= 4 and np.isfinite(box[0]):
            xs.append(float((box[0] + box[2]) / 2.0))
            ys.append(float((box[1] + box[3]) / 2.0))
        else:
            xs.append(None)
            ys.append(None)
    return xs, ys


def _geom_bounces(xs, ys, min_gap: int = 8) -> List[int]:
    n = len(xs)
    cand = []
    for i in range(2, n - 2):
        if any(xs[j] is None or ys[j] is None for j in range(i - 2, i + 3)):
            continue
        y0, y1, y2 = ys[i - 1], ys[i], ys[i + 1]
        if (y1 - y0) * (y2 - y1) < 0:
            cand.append(i)
    keep = []
    for i in cand:
        if not keep or i - keep[-1] >= min_gap:
            keep.append(i)
        else:
            prev = keep[-1]
            score_i = abs(ys[i + 1] - ys[i - 1])
            score_p = abs(ys[prev + 1] - ys[prev - 1])
            if score_i > score_p:
                keep[-1] = i
    return keep


def _catboost_bounces(xs, ys, model_path: str, threshold: float = 0.45) -> List[int]:
    import pandas as pd
    from catboost import CatBoostRegressor
    from scipy.interpolate import CubicSpline
    from scipy.spatial import distance

    x_ball = list(xs)
    y_ball = list(ys)
    is_none = [int(x is None) for x in x_ball]
    interp = 5
    counter = 0
    for num in range(interp, len(x_ball) - 1):
        if not x_ball[num] and sum(is_none[num - interp : num]) == 0 and counter < 3:
            xs_w = list(range(interp))
            fx = CubicSpline(xs_w, x_ball[num - interp : num], bc_type="natural")
            fy = CubicSpline(xs_w, y_ball[num - interp : num], bc_type="natural")
            x_ext, y_ext = float(fx(interp)), float(fy(interp))
            x_ball[num] = x_ext
            y_ball[num] = y_ext
            is_none[num] = 0
            if x_ball[num + 1]:
                if distance.euclidean((x_ext, y_ext), (x_ball[num + 1], y_ball[num + 1])) > 80:
                    x_ball[num + 1], y_ball[num + 1], is_none[num + 1] = None, None, 1
            counter += 1
        else:
            counter = 0

    labels = pd.DataFrame({"frame": range(len(x_ball)), "x-coordinate": x_ball, "y-coordinate": y_ball})
    eps = 1e-15
    for i in range(1, 3):
        labels[f"x_lag_{i}"] = labels["x-coordinate"].shift(i)
        labels[f"x_lag_inv_{i}"] = labels["x-coordinate"].shift(-i)
        labels[f"y_lag_{i}"] = labels["y-coordinate"].shift(i)
        labels[f"y_lag_inv_{i}"] = labels["y-coordinate"].shift(-i)
        labels[f"x_diff_{i}"] = abs(labels[f"x_lag_{i}"] - labels["x-coordinate"])
        labels[f"y_diff_{i}"] = labels[f"y_lag_{i}"] - labels["y-coordinate"]
        labels[f"x_diff_inv_{i}"] = abs(labels[f"x_lag_inv_{i}"] - labels["x-coordinate"])
        labels[f"y_diff_inv_{i}"] = labels[f"y_lag_inv_{i}"] - labels["y-coordinate"]
        labels[f"x_div_{i}"] = abs(labels[f"x_diff_{i}"] / (labels[f"x_diff_inv_{i}"] + eps))
        labels[f"y_div_{i}"] = labels[f"y_diff_{i}"] / (labels[f"y_diff_inv_{i}"] + eps)
    for i in range(1, 3):
        labels = labels[labels[f"x_lag_{i}"].notna()]
        labels = labels[labels[f"x_lag_inv_{i}"].notna()]
        labels = labels[labels["x-coordinate"].notna()]
    colnames = (
        [f"x_diff_{i}" for i in range(1, 3)]
        + [f"x_diff_inv_{i}" for i in range(1, 3)]
        + [f"x_div_{i}" for i in range(1, 3)]
        + [f"y_diff_{i}" for i in range(1, 3)]
        + [f"y_diff_inv_{i}" for i in range(1, 3)]
        + [f"y_div_{i}" for i in range(1, 3)]
    )
    features = labels[colnames]
    frames = list(labels["frame"])
    model = CatBoostRegressor()
    model.load_model(model_path)
    preds = model.predict(features)
    ind = np.where(preds > threshold)[0]
    if len(ind) == 0:
        return []
    filtered = [int(ind[0])]
    for i in range(1, len(ind)):
        if ind[i] - ind[i - 1] != 1:
            filtered.append(int(ind[i]))
        elif preds[ind[i]] > preds[ind[i - 1]]:
            filtered[-1] = int(ind[i])
    return [int(frames[i]) for i in filtered]


def default_bounce_weights() -> Optional[str]:
    for p in (
        "high_angle_dataset/ctb_regr_bounce.cbm",
        "weights/ctb_regr_bounce.cbm",
        os.path.join(os.path.dirname(__file__), "..", "..", "high_angle_dataset", "ctb_regr_bounce.cbm"),
    ):
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None


def detect_bounces(filled, model_path: Optional[str] = None) -> List[int]:
    xs, ys = _centers(filled)
    path = model_path or default_bounce_weights()
    if path:
        try:
            return _catboost_bounces(xs, ys, path)
        except Exception:
            pass
    return _geom_bounces(xs, ys)
