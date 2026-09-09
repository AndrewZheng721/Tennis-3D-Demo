from typing import Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .court3d import COURT_LINES_3D, COURT_XYZ, DB, NET


def _court(ax):
    for a, b in COURT_LINES_3D:
        p, q = COURT_XYZ[a], COURT_XYZ[b]
        ax.plot([p[0], q[0]], [p[1], q[1]], [0, 0], color="0.35", lw=1.0)
    ax.plot([-DB, DB], [0, 0], [0, 0], color="0.5", lw=0.8)
    ax.plot([-DB, -DB], [0, 0], [0, 1.07], color="0.45", lw=1.0)
    ax.plot([DB, DB], [0, 0], [0, 1.07], color="0.45", lw=1.0)
    ax.plot([-DB, DB], [0, 0], [1.07, 1.07], color="0.45", lw=1.0)


def _axes(ax):
    ax.quiver(0, 0, 0, 3.2, 0, 0, color="#c0392b", arrow_length_ratio=0.12, lw=2.0)
    ax.quiver(0, 0, 0, 0, 3.2, 0, color="#1e8449", arrow_length_ratio=0.12, lw=2.0)
    ax.quiver(0, 0, 0, 0, 0, 3.2, color="#2471a3", arrow_length_ratio=0.12, lw=2.0)
    ax.text(3.5, 0, 0, "X", color="#c0392b", fontsize=11)
    ax.text(0, 3.5, 0, "Y  far", color="#1e8449", fontsize=11)
    ax.text(0, 0, 3.5, "Z", color="#2471a3", fontsize=11)


def _setup(ax):
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim(-7, 7)
    ax.set_ylim(-13, 13)
    ax.set_zlim(0, 6)
    try:
        ax.set_box_aspect((14, 26, 6))
    except Exception:
        pass
    ax.view_init(elev=22, azim=-72)
    ax.grid(True, alpha=0.25)


def _fig_bgr(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)


def write_traj3d_video(traj3d, path: str, preview_path: Optional[str] = None):
    fps = float(traj3d.get("fps") or 30)
    recs = traj3d["frames"]
    fig = plt.figure(figsize=(10.24, 7.68), dpi=100, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)
    h0, w0, _ = _fig_bgr(fig).shape
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w0, h0))
    trail = []
    first = True
    for rec in tqdm(recs, desc="write 3d"):
        ax.cla()
        _setup(ax)
        _court(ax)
        _axes(ax)
        xyz = rec.get("xyz")
        if xyz:
            trail.append(xyz)
            xs, ys, zs = zip(*trail[-80:])
            ax.plot(xs, ys, zs, color="#1abc9c", lw=2.0)
            ax.scatter([xyz[0]], [xyz[1]], [0.0], c="#7f8c8d", s=18)
            ax.plot([xyz[0], xyz[0]], [xyz[1], xyz[1]], [0, xyz[2]], color="#95a5a6", lw=0.8)
            ax.scatter([xyz[0]], [xyz[1]], [xyz[2]], c="#e67e22", s=36)
            ax.set_title(
                f"frame {rec['frame_id']}   X={xyz[0]:.2f}  Y={xyz[1]:.2f}  Z={xyz[2]:.2f} m",
                fontsize=11,
            )
        else:
            trail = []
            ax.set_title(f"frame {rec['frame_id']}   no 3D", fontsize=11)
        img = _fig_bgr(fig)
        if img.shape[1] != w0 or img.shape[0] != h0:
            img = cv2.resize(img, (w0, h0))
        if first and preview_path:
            cv2.imwrite(preview_path, img)
            first = False
        writer.write(img)
    writer.release()
    plt.close(fig)
