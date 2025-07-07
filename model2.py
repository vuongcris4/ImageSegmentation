import torch.nn as nn
import torch.nn.functional as F
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        x = F.relu(x)
        return x

class myModel(nn.Module):
    def __init__(self, n_classes):
        super(myModel, self).__init__()
        # Encoder
        self.encoder1 = ResidualBlock(3, 8)    # 3 -> 8 channels
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = ResidualBlock(8, 16)   # 8 -> 16 channels
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = ResidualBlock(16, 32)  # 16 -> 32 channels
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(32, 64)  # 32 -> 64 channels

        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.decoder3 = ResidualBlock(64 + 32, 32)  # 64 (bottleneck) + 32 (encoder3) -> 32
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.decoder2 = ResidualBlock(32 + 16, 16)  # 32 (decoder3) + 16 (encoder2) -> 16
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.decoder1 = ResidualBlock(16 + 8, 8)    # 16 (decoder2) + 8 (encoder1) -> 8

        # Output layer
        self.conv_out = nn.Conv2d(8, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1 = self.encoder1(x)
        x2 = self.pool1(x1)
        x2 = self.encoder2(x2)
        x3 = self.pool2(x2)
        x3 = self.encoder3(x3)
        x4 = self.pool3(x3)
        x4 = self.bottleneck(x4)

        # Decoder path with skip connections
        x = self.up3(x4)
        x = torch.cat([x, x3], dim=1)  # Skip connection from encoder3
        x = self.decoder3(x)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)  # Skip connection from encoder2
        x = self.decoder2(x)
        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)  # Skip connection from encoder1
        x = self.decoder1(x)
        x = self.conv_out(x)

        return F.softmax(x, dim=1)

# Để kiểm tra số tham số:
model = myModel(n_classes=3)
from utils import count_parameters
count_parameters(model)


