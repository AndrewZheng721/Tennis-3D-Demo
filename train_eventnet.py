import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.score.eventnet import FEAT, WIN, EventNet, WindowSet, build_split, class_weight


def _find_data():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "high_angle_dataset", "events"))


def _scores(pred, gt):
    out = {}
    for c, name in ((1, "hit"), (2, "bounce")):
        tp = int(np.sum((pred == c) & (gt == c)))
        fp = int(np.sum((pred == c) & (gt != c)))
        fn = int(np.sum((pred != c) & (gt == c)))
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        out[name] = (p, r, 2 * p * r / (p + r + 1e-9), tp, fp, fn)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None)
    p.add_argument("--out", default="weights/eventnet.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--window", type=int, default=WIN)
    return p.parse_args()


def main():
    args = parse_args()
    root = args.data or _find_data()
    if not os.path.isdir(root):
        raise SystemExit(f"data not found: {root}")
    (xtr, ytr), (xva, yva), val_games = build_split(root, args.window)
    print("data", root)
    print("train", len(ytr), "val", len(yva), "val_games", val_games)
    counts = {c: int(np.sum(ytr == c)) for c in (0, 1, 2)}
    print("train status", counts)
    if counts[1] == 0 or counts[2] == 0:
        raise SystemExit("events 里还没有击球/落地标注。先改 Label.csv 的 status：1 击球，2 落地。")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(WindowSet(xtr, ytr), batch_size=args.batch, shuffle=True, drop_last=False)
    val_loader = DataLoader(WindowSet(xva, yva), batch_size=args.batch, shuffle=False) if len(yva) else None
    model = EventNet(in_dim=FEAT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = torch.nn.CrossEntropyLoss(weight=class_weight(ytr).to(device))
    best = -1.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        tr = 0.0
        n = 0
        for xb, yb in tqdm(train_loader, desc=f"train {epoch}"):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tr += float(loss.item())
            n += 1
        tr /= max(n, 1)
        if val_loader is None:
            print(f"epoch {epoch} train {tr:.4f}")
            score = tr
            better = epoch == args.epochs - 1
        else:
            model.eval()
            preds, gts = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred = model(xb.to(device)).argmax(1).cpu().numpy()
                    preds.append(pred)
                    gts.append(yb.numpy())
            pred = np.concatenate(preds)
            gt = np.concatenate(gts)
            sc = _scores(pred, gt)
            score = 0.5 * (sc["hit"][2] + sc["bounce"][2])
            better = score > best
            print(
                f"epoch {epoch} train {tr:.4f} "
                f"hit_f1 {sc['hit'][2]:.3f} bounce_f1 {sc['bounce'][2]:.3f} "
                f"bounce_tp {sc['bounce'][3]} fp {sc['bounce'][4]} fn {sc['bounce'][5]}"
            )
        if better:
            best = score
            torch.save(
                {
                    "state": model.state_dict(),
                    "in_dim": FEAT,
                    "hidden": model.hidden,
                    "layers": model.layers,
                    "window": args.window,
                },
                args.out,
            )
            print("saved", args.out)


if __name__ == "__main__":
    main()
