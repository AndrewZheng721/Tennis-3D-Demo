import os
import cv2
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse

import json

from functools import partial

# -----------------------------
# 1. COCO → H36M
# -----------------------------
def coco17_to_h36m17(kp17):
    out = np.zeros((17,3), dtype=np.float32)

    pelvis = (kp17[11] + kp17[12]) / 2
    thorax = (kp17[5] + kp17[6]) / 2

    out[0]  = pelvis
    out[1]  = kp17[12]
    out[2]  = kp17[14]
    out[3]  = kp17[16]
    out[4]  = kp17[11]
    out[5]  = kp17[13]
    out[6]  = kp17[15]
    out[7]  = (pelvis + thorax) / 2
    out[8]  = thorax
    out[9]  = kp17[0]
    out[10] = kp17[0]
    out[11] = kp17[5]
    out[12] = kp17[7]
    out[13] = kp17[9]
    out[14] = kp17[6]
    out[15] = kp17[8]
    out[16] = kp17[10]

    return out

def pose3d_to_json(pose3d_dict):
    json_data = []

    for tid, seq in pose3d_dict.items():

        track_obj = {
            "track_id": int(tid),
            "frames": []
        }

        for i, frame in enumerate(seq):

            track_obj["frames"].append({
                "frame_id": int(i),
                "joints3d": frame.tolist()
            })

        json_data.append(track_obj)

    return json_data

# -----------------------------
# 2. Pose3D Visualizer
# -----------------------------
class Pose3DVisualizer:

    H36M_BONES = [
        (0,1),(1,2),(2,3),
        (0,4),(4,5),(5,6),
        (0,7),(7,8),(8,9),(9,10),
        (8,11),(11,12),(12,13),
        (8,14),(14,15),(15,16)
    ]

    def __init__(self):
        self.fig = plt.figure(figsize=(6,6))
        self.ax = self.fig.add_subplot(111, projection="3d")

    def draw(self, joints3d):

        self.ax.clear()

        x = joints3d[:,0]
        y = joints3d[:,2]
        z = -joints3d[:,1]

        self.ax.scatter(x, y, z, s=10)

        for s,e in self.H36M_BONES:
            self.ax.plot([x[s], x[e]],
                         [y[s], y[e]],
                         [z[s], z[e]])

        self.ax.set_xlim(-1,1)
        self.ax.set_ylim(-1,1)
        self.ax.set_zlim(-1,1)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.fig.canvas.draw()

        img = np.asarray(self.fig.canvas.buffer_rgba())
        return img[:,:,:3].copy()

parser = argparse.ArgumentParser(description="第二阶段：3D姿态估计脚本")
parser.add_argument("--input", type=str, help="输入 tracked_pose.pkl 路径")
parser.add_argument("--out", type=str, help="输出目录路径")
args = parser.parse_args()
# -----------------------------
# 3. Load Stage1
# -----------------------------
tracked_pkl = args.input

with open(tracked_pkl, "rb") as f:
    frames = pickle.load(f)

track_dict = {}

for frame in frames:
    for player in frame.players:

        kp = player.keypoints

        if kp.shape[1] == 2:
            conf = np.ones((17,1))
            kp = np.concatenate([kp, conf], axis=1)

        kp_h36m = coco17_to_h36m17(kp)

        tid = int(player.track_id)

        if tid not in track_dict:
            track_dict[tid] = []

        track_dict[tid].append(kp_h36m)

count = 0
for tid in track_dict:
    seq = np.stack(track_dict[tid], axis=0)  # (T,17,3)

    # -----------------------------
    # 归一化 x,y 到 [-1,1]
    # -----------------------------
    xy = seq[:, :, :2]
    xmin = xy[:, :, 0].min()
    xmax = xy[:, :, 0].max()
    ymin = xy[:, :, 1].min()
    ymax = xy[:, :, 1].max()

    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    scale = max(xmax - xmin, ymax - ymin) / 2.0  # [-1,1]归一化

    seq[:, :, 0] = (seq[:, :, 0] - center_x) / scale
    seq[:, :, 1] = (seq[:, :, 1] - center_y) / scale

    track_dict[tid] = seq.astype(np.float32)

    print(count, "mean:", seq.mean())
    print(count, "std:", seq.std())
    print(count, "min:", seq.min(), "max:", seq.max())


# -----------------------------
# 4. Load MotionBERT
# -----------------------------
import sys
sys.path.append("MotionBERT")

from lib.utils.tools import get_config
from lib.model.DSTformer import DSTformer

cfg = get_config("MotionBERT/configs/pose3d/MB_ft_h36m_global_lite.yaml")

model_pos = DSTformer(
    dim_in=3,
    dim_out=3,
    dim_feat=cfg.dim_feat,
    dim_rep=cfg.dim_rep,
    depth=cfg.depth,
    num_heads=cfg.num_heads,
    mlp_ratio=cfg.mlp_ratio,
    maxlen=cfg.maxlen,
    num_joints=cfg.num_joints,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    att_fuse=cfg.att_fuse
)

model_pos = model_pos.cuda()
model_pos = nn.DataParallel(model_pos)

checkpoint = torch.load(
    "MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin"
)

model_pos.load_state_dict(checkpoint["model_pos"])
model_pos.eval()


# -----------------------------
# 5. Inference + Visualization
# -----------------------------

visualizer = Pose3DVisualizer()

writer = cv2.VideoWriter(
    os.path.join(args.out, "pose3d_vis.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (600,600)
)

pose3d_dict = {}

WINDOW = 81
STRIDE = 27

with torch.no_grad():

    for tid, seq in track_dict.items():

        seq = np.stack(seq)

        if len(seq) < WINDOW:
            continue

        all_frames = np.zeros_like(seq)

        count = np.zeros((len(seq),1))

        for start in range(0, len(seq)-WINDOW+1, STRIDE):

            clip = seq[start:start+WINDOW]

            x = torch.from_numpy(clip[None]).float().cuda()

            pred = model_pos(x)[0].cpu().numpy()

            for i in range(WINDOW):

                idx = start + i
                all_frames[idx] += pred[i]
                count[idx] += 1

        valid = count[:,0] > 0

        if np.any(valid):
            all_frames[valid] /= count[valid][:, None]

        pose3d_dict[tid] = all_frames

        # -----------------------------
        # 6. Visualization
        # -----------------------------
        for frame3d in all_frames:

            img = visualizer.draw(frame3d)
            img = cv2.resize(img, (600,600))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            writer.write(img)

writer.release()

with open(os.path.join(args.out, "pose3d.pkl"), "wb") as f:
    pickle.dump(pose3d_dict, f)

json_data = pose3d_to_json(pose3d_dict)

with open(os.path.join(args.out, "pose3d.json"), "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=2)

print("Done: pose3d + visualization saved")