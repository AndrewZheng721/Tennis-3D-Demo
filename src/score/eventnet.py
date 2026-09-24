import json
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


WIN = 21
FEAT = 12


class EventNet(nn.Module):
    def __init__(self, in_dim: int = FEAT, hidden: int = 64, layers: int = 2, n_cls: int = 3):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.layers = layers
        self.proj = nn.Linear(in_dim, 32)
        self.gru = nn.GRU(32, hidden, layers, batch_first=True, bidirectional=True, dropout=0.2)
        self.head = nn.Linear(hidden * 2, n_cls)

    def forward(self, x):
        h = torch.relu(self.proj(x))
        out, _ = self.gru(h)
        mid = out.size(1) // 2
        return self.head(out[:, mid])


def _scale(x, seen):
    if seen.any():
        return max(float(np.max(x[seen])), 1.0)
    return 1.0


def clip_arrays(df: pd.DataFrame, players=None) -> Tuple[np.ndarray, np.ndarray]:
    vis = pd.to_numeric(df["visibility"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    x = pd.to_numeric(df["x-coordinate"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    y = pd.to_numeric(df["y-coordinate"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    st = pd.to_numeric(df["status"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    seen = vis > 0
    sx, sy = _scale(x, seen), _scale(y, seen)
    xn = np.zeros_like(x)
    yn = np.zeros_like(y)
    xn[seen] = x[seen] / sx
    yn[seen] = y[seen] / sy
    prev = np.zeros_like(seen)
    prev[1:] = seen[:-1]
    ok = seen & prev
    dx = np.zeros_like(xn)
    dy = np.zeros_like(yn)
    dx[1:] = xn[1:] - xn[:-1]
    dy[1:] = yn[1:] - yn[:-1]
    dx[~ok] = 0
    dy[~ok] = 0
    n = len(df)
    near = np.zeros((n, 2), np.float32)
    far = np.zeros((n, 2), np.float32)
    has = np.zeros(n, np.float32)
    if players:
        for fr in players.get("frames", []):
            i = int(fr.get("frame_id", -1))
            if i < 0 or i >= n:
                continue
            got = False
            for p in fr.get("players") or []:
                b = p.get("bbox") or []
                if len(b) < 4:
                    continue
                pt = (float(b[0] + b[2]) * 0.5 / sx, float(b[3]) / sy)
                if p.get("side") == "far":
                    far[i] = pt
                    got = True
                elif p.get("side") == "near":
                    near[i] = pt
                    got = True
            if got:
                has[i] = 1.0
    dist_n = np.hypot(xn - near[:, 0], yn - near[:, 1]) * has
    dist_f = np.hypot(xn - far[:, 0], yn - far[:, 1]) * has
    feat = np.stack(
        [
            xn,
            yn,
            dx,
            dy,
            seen.astype(np.float32),
            near[:, 0],
            near[:, 1],
            far[:, 0],
            far[:, 1],
            dist_n,
            dist_f,
            has,
        ],
        axis=1,
    ).astype(np.float32)
    st = np.where(np.isin(st, (0, 1, 2)), st, 0).astype(np.int64)
    return feat, st


def windows(feat: np.ndarray, st: np.ndarray, win: int = WIN):
    half = win // 2
    pad = np.zeros((half, feat.shape[1]), np.float32)
    f = np.concatenate([pad, feat, pad], axis=0)
    xs = np.stack([f[i : i + win] for i in range(len(st))]).astype(np.float32)
    return xs, st


def load_players(clip_dir: str):
    path = os.path.join(clip_dir, "players.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iter_clips(root: str):
    for dirpath, _, files in os.walk(root):
        if "Label.csv" not in files:
            continue
        yield dirpath, os.path.join(dirpath, "Label.csv")


def game_of(clip_dir: str) -> str:
    return os.path.basename(os.path.dirname(clip_dir))


def build_split(root: str, win: int = WIN, val_ratio: float = 0.2):
    clips = []
    for clip_dir, csv_path in iter_clips(root):
        df = pd.read_csv(csv_path)
        if "tennis_high_angle_01" in clip_dir:
            continue
        if len(df) < 5 or "status" not in df.columns:
            continue
        feat, st = clip_arrays(df, load_players(clip_dir))
        xs, ys = windows(feat, st, win)
        clips.append((game_of(clip_dir), xs, ys))
    games = sorted({g for g, _, _ in clips})
    n_val = max(1, int(round(len(games) * val_ratio))) if len(games) > 1 else 0
    val_games = set(games[-n_val:]) if n_val else set()
    tr_x, tr_y, va_x, va_y = [], [], [], []
    for g, xs, ys in clips:
        if g in val_games:
            va_x.append(xs)
            va_y.append(ys)
        else:
            tr_x.append(xs)
            tr_y.append(ys)
    def cat(xs, ys):
        if not xs:
            return np.zeros((0, win, FEAT), np.float32), np.zeros((0,), np.int64)
        return np.concatenate(xs, 0), np.concatenate(ys, 0)
    return cat(tr_x, tr_y), cat(va_x, va_y), sorted(val_games)


class WindowSet(torch.utils.data.Dataset):
    def __init__(self, xs, ys):
        self.xs = torch.from_numpy(xs)
        self.ys = torch.from_numpy(ys)

    def __len__(self):
        return int(self.ys.shape[0])

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


def class_weight(ys: np.ndarray) -> torch.Tensor:
    w = []
    n = max(len(ys), 1)
    for c in range(3):
        k = int(np.sum(ys == c))
        w.append(n / (3.0 * max(k, 1)))
    return torch.tensor(w, dtype=torch.float32)


def load_eventnet(path: str, device=None) -> EventNet:
    ckpt = torch.load(path, map_location=device or "cpu")
    model = EventNet(
        in_dim=int(ckpt.get("in_dim", FEAT)),
        hidden=int(ckpt.get("hidden", 64)),
        layers=int(ckpt.get("layers", 2)),
    )
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model


@torch.no_grad()
def predict_status(model: EventNet, df: pd.DataFrame, players=None, device=None, win: int = WIN) -> np.ndarray:
    device = device or next(model.parameters()).device
    feat, _ = clip_arrays(df, players)
    xs, _ = windows(feat, np.zeros(len(df), np.int64), win)
    model.eval()
    out = []
    for i in range(0, len(xs), 256):
        batch = torch.from_numpy(xs[i : i + 256]).to(device)
        pred = model(batch).argmax(dim=1).cpu().numpy()
        out.append(pred)
    if not out:
        return np.zeros((0,), np.int64)
    return np.concatenate(out, 0)
