import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PoseActionDataset
from model import STGCN_Like


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = PoseActionDataset("outputs/pose3d.pkl")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = STGCN_Like(num_classes=4).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20

    for epoch in range(epochs):
        total_loss = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)

            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: loss = {total_loss:.4f}")

    torch.save(model.state_dict(), "stage3_action_model.pth")
    print("Saved model.")


if __name__ == "__main__":
    train()