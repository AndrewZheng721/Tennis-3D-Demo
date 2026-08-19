import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


LABEL_MAP = {
    "serve": 0,
    'forehand_topspin': 1,
    'forehand_slice': 2,
    'backhand': 3,
    'volley': 4,
    'smash': 5,
    'other': 6,
    'ready': 7
}


class PoseActionDataset(Dataset):
    """
    Input:
        (T,17,3)
    Output:
        (64,17,3), label
    """

    def __init__(self, pkl_path, window=64, stride=16):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.window = window
        self.stride = stride

        self.samples = []

        seq = self._extract_seq(data)

        clips = self._sliding_window(seq)

        # ⚠️ 这里先用 dummy label（后面你换成真实标注）
        for i, clip in enumerate(clips):
            label = i % len(LABEL_MAP)
            self.samples.append((clip, label))

    def _extract_seq(self, data):
        seq = []
        for frame in data:
            if len(frame.players) == 0:
                continue
            kp = frame.players[0].keypoints  # (17,3)
            seq.append(kp)

        return np.array(seq)

    def _sliding_window(self, seq):
        clips = []
        T = len(seq)

        for i in range(0, T - self.window, self.stride):
            clip = seq[i:i + self.window]
            clips.append(clip)

        return clips

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x, y