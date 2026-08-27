import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.ball.tracknet_models import TrackNetV1
from src.ball.tracknet_tracker import _load_state


class TrackNetCsv(Dataset):
    def __init__(self, root, csv_name):
        self.root = root
        self.data = pd.read_csv(os.path.join(root, csv_name))

    def __len__(self):
        return len(self.data)

    def _read(self, rel):
        img = cv2.imread(os.path.join(self.root, rel))
        if img is None:
            raise FileNotFoundError(rel)
        img = cv2.resize(img, (640, 360))
        return img.astype(np.float32) / 255.0

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        imgs = np.concatenate(
            [self._read(row.path1), self._read(row.path2), self._read(row.path3)], axis=2
        )
        inp = np.transpose(imgs, (2, 0, 1)).copy()
        hm = np.zeros((360, 640), dtype=np.float32)
        if int(row.vis) != 0:
            x, y = float(row.x), float(row.y)
            yy, xx = np.ogrid[:360, :640]
            g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 10.0)
            hm = g / max(float(g.max()), 1e-6) * 255.0
        gt = hm.astype(np.int64).reshape(-1)
        return torch.from_numpy(inp), torch.from_numpy(gt)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset/tracknet_auto")
    p.add_argument("--weights", default="weights/tracknet.pth")
    p.add_argument("--out", default="weights/tracknet_ft.pth")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = TrackNetCsv(args.data, "labels_train.csv")
    val_set = TrackNetCsv(args.data, "labels_val.csv")
    train_loader = DataLoader(
        train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True
    )
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    model = TrackNetV1()
    model.load_state_dict(_load_state(args.weights), strict=True)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    best = 1e9
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        tr = 0.0
        n = 0
        for inp, gt in tqdm(train_loader, desc=f"train {epoch}"):
            inp, gt = inp.to(device), gt.to(device)
            opt.zero_grad()
            out = model(inp)
            loss = crit(out, gt)
            loss.backward()
            opt.step()
            tr += float(loss.item())
            n += 1
        model.eval()
        va = 0.0
        m = 0
        with torch.no_grad():
            for inp, gt in val_loader:
                inp = inp.to(device)
                gt = gt.to(device)
                va += float(crit(model(inp), gt).item())
                m += 1
        tr /= max(n, 1)
        va /= max(m, 1)
        print(f"epoch {epoch} train {tr:.4f} val {va:.4f}")
        if va < best:
            best = va
            torch.save(model.state_dict(), args.out)
            print("saved", args.out)


if __name__ == "__main__":
    main()
