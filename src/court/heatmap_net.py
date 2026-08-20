import torch
import torch.nn as nn


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


class CourtHeatmapNet(nn.Module):
    """TennisCourtDetector 的网络：输入一张 640×360 的图，输出 15 张热力图。

    热力图可以想成一张「概率地图」。第 k 张图上某个像素越亮，
    越表示「第 k 个球场角点就在这里」。
    比 ResNet 直接报一个 (x,y) 更抗透视变化，因为每个点是在整张图上找亮斑。
    第 15 张是球场中心，训练时用来帮忙收敛，推理只用前 14 张。
    """

    def __init__(self, out_channels=15):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = ConvBlock(3, 64)
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
        x = self.conv2(self.conv1(x))
        x = self.pool1(x)
        x = self.conv4(self.conv3(x))
        x = self.pool2(x)
        x = self.conv7(self.conv6(self.conv5(x)))
        x = self.pool3(x)
        x = self.conv10(self.conv9(self.conv8(x)))
        x = self.conv13(self.conv12(self.conv11(self.ups1(x))))
        x = self.conv15(self.conv14(self.ups2(x)))
        x = self.conv18(self.conv17(self.conv16(self.ups3(x))))
        return x
