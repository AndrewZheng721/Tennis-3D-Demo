"""TrackNet 系列网络结构。

V1：yastrebksv 非官方实现，输入 3 帧 640×360，输出 256 档灰度热力图。
    网球转播有现成权重，高位小球比 YOLO 稳得多。

V4 Type A：官方 TrackNetV2 骨架 + 帧差运动注意力（ICASSP 2025）。
    输入 3 帧 512×288，输出 3 张 0～1 热力图。
    官方权重是 TensorFlow .keras；这里接 PyTorch 的 .pth。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvReluBn(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        pad = 0 if kernel_size == 1 else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class TrackNetV1(nn.Module):
    """网球 TrackNet V1，和 yastrebksv/TrackNet 权重一一对应。"""

    def __init__(self, out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = ConvBlock(9, 64)
        self.conv2 = ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv8 = ConvBlock(256, 512)
        self.conv9 = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(512, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(256, 128)
        self.conv15 = ConvBlock(128, 128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(128, 64)
        self.conv17 = ConvBlock(64, 64)
        self.conv18 = ConvBlock(64, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        b = x.size(0)
        return x.reshape(b, self.out_channels, -1)


class MotionPrompt(nn.Module):
    """官方 MotionPromptLayer 的简化版：灰度帧差 → 可学习门控。"""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.1))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x9):
        b = x9.size(0)
        frames = x9.view(b, 3, 3, x9.size(2), x9.size(3))
        gray = (
            0.299 * frames[:, :, 0]
            + 0.587 * frames[:, :, 1]
            + 0.114 * frames[:, :, 2]
        )
        diff = gray[:, 1:] - gray[:, :-1]
        scale = 5.0 / (0.45 * torch.abs(torch.tanh(self.a)) + 0.1)
        attn = torch.sigmoid(scale * (diff.abs() - 0.6 * torch.tanh(self.b)))
        return attn


class FusionTypeA(nn.Module):
    def forward(self, visual, attn):
        return torch.stack(
            [
                visual[:, 0],
                visual[:, 1] * attn[:, 0],
                visual[:, 2] * attn[:, 1],
            ],
            dim=1,
        )


class TrackNetV4(nn.Module):
    """官方 TrackNetV4 Type A：V2 U-Net + 末端运动融合。输入 9×288×512。"""

    def __init__(self):
        super().__init__()
        self.motion = MotionPrompt()
        self.fusion = FusionTypeA()
        self.c1 = _ConvReluBn(9, 64)
        self.c2 = _ConvReluBn(64, 64)
        self.c4 = _ConvReluBn(64, 128)
        self.c5 = _ConvReluBn(128, 128)
        self.c7 = _ConvReluBn(128, 256)
        self.c8 = _ConvReluBn(256, 256)
        self.c9 = _ConvReluBn(256, 256)
        self.c11 = _ConvReluBn(256, 512)
        self.c12 = _ConvReluBn(512, 512)
        self.c13 = _ConvReluBn(512, 512)
        self.c15 = _ConvReluBn(768, 256)
        self.c16 = _ConvReluBn(256, 256)
        self.c17 = _ConvReluBn(256, 256)
        self.c19 = _ConvReluBn(384, 128)
        self.c20 = _ConvReluBn(128, 128)
        self.c22 = _ConvReluBn(192, 64)
        self.c23 = _ConvReluBn(64, 64)
        self.c24 = nn.Conv2d(64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        attn = self.motion(x)
        x1 = self.c2(self.c1(x))
        x2 = self.c5(self.c4(self.pool(x1)))
        x3 = self.c9(self.c8(self.c7(self.pool(x2))))
        x = self.c13(self.c12(self.c11(self.pool(x3))))
        x = self.c17(self.c16(self.c15(torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), x3], 1))))
        x = self.c20(self.c19(torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), x2], 1)))
        x = self.c23(self.c22(torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), x1], 1)))
        visual = self.c24(x)
        return torch.sigmoid(self.fusion(visual, attn))
