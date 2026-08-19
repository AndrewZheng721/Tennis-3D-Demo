import torch


class MotionBERTLoader:

    def __init__(
        self,
        model_path
    ):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        ckpt = torch.load(
            model_path,
            map_location=self.device
        )

        self.model = ckpt["model"]

        self.model.eval()

        self.model.to(
            self.device
        )