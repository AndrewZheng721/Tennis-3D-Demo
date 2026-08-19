import torch
import numpy as np

class Pose3DInference:
    def __init__(self, model):
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def infer(self, seq2d):
        """
        seq2d: (T,17,2)
        返回 (T,17,3)
        """
        with torch.no_grad():
            x = torch.tensor(seq2d, dtype=torch.float32).unsqueeze(0).to(self.device)
            pred = self.model(x)
            pred = pred.squeeze(0).cpu().numpy()
        return pred