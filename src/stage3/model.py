import torch
import torch.nn as nn


class STGCN_Like(nn.Module):
    ## baseline

    def __init__(self, num_joints=17, num_classes=4):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(128 * num_joints, num_classes)

    def forward(self, x):
        """
        x: (B, T, J, 3)
        """

        B, T, J, C = x.shape

        x = x.permute(0, 3, 1, 2)  # (B,3,T,J)

        x = self.conv1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = torch.relu(x)

        x = self.pool(x)  # (B,128,1,1)

        x = x.view(B, -1)

        x = self.fc(x)

        return x