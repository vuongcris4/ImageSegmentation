import torch
import torch.nn as nn
import torch.nn.functional as F

# Định nghĩa ChannelAttention
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Định nghĩa ResidualBlock với tùy chọn Dilated Convolution
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ResidualBlock, self).__init__()
        # Lớp tích chập 5x5 với padding điều chỉnh theo dilation
        padding = 2 * dilation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=padding, stride=1, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=padding, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out += identity
        out = F.relu(self.bn2(out))
        return out

# Định nghĩa mô hình myModel cải tiến
class myModel(nn.Module):
    def __init__(self, n_classes, dropout_prob=0.2):
        super(myModel, self).__init__()
        
        # --- Encoder ---
        self.encoder1 = ResidualBlock(3, 8)             # 3 -> 8 kênh
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ResidualBlock(8, 16)            # 8 -> 16 kênh
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ResidualBlock(16, 32)           # 16 -> 32 kênh
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ResidualBlock(32, 64, dilation=2)  # 32 -> 64 kênh, thêm dilation
        self.channel_attention = ChannelAttention(64)   # Thêm Channel Attention

        # --- Skip connections với Conv 1x1 và BatchNorm ---
        self.conv_skip3 = nn.Conv2d(32, 32, kernel_size=1)
        self.bn_skip3 = nn.BatchNorm2d(32)
        self.conv_skip2 = nn.Conv2d(16, 16, kernel_size=1)
        self.bn_skip2 = nn.BatchNorm2d(16)
        self.conv_skip1 = nn.Conv2d(8, 8, kernel_size=1)
        self.bn_skip1 = nn.BatchNorm2d(8)

        # --- Dropout ---
        self.dropout_bottleneck = nn.Dropout2d(p=dropout_prob)
        self.dropout3 = nn.Dropout2d(p=dropout_prob)
        self.dropout2 = nn.Dropout2d(p=dropout_prob)
        self.dropout1 = nn.Dropout2d(p=dropout_prob)

        # --- Decoder ---
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn_cat3 = nn.BatchNorm2d(96)  # 64 + 32
        self.conv_cat3 = nn.Conv2d(64 + 32, 32, kernel_size=1)
        self.decoder3 = ResidualBlock(32, 32)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn_cat2 = nn.BatchNorm2d(48)  # 32 + 16
        self.conv_cat2 = nn.Conv2d(32 + 16, 16, kernel_size=1)
        self.decoder2 = ResidualBlock(16, 16)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn_cat1 = nn.BatchNorm2d(24)  # 16 + 8
        self.conv_cat1 = nn.Conv2d(16 + 8, 8, kernel_size=1)
        self.decoder1 = ResidualBlock(8, 8)

        # --- Output layer ---
        self.conv_out = nn.Conv2d(8, n_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder path ---
        x1 = self.encoder1(x)           # (B, 8, H, W)
        p1 = self.pool1(x1)             # (B, 8, H/2, W/2)
        x2 = self.encoder2(p1)          # (B, 16, H/2, W/2)
        p2 = self.pool2(x2)             # (B, 16, H/4, W/4)
        x3 = self.encoder3(p2)          # (B, 32, H/4, W/4)
        p3 = self.pool3(x3)             # (B, 32, H/8, W/8)
        b = self.bottleneck(p3)         # (B, 64, H/8, W/8)
        b = self.channel_attention(b)   # Áp dụng Channel Attention
        b = self.dropout_bottleneck(b)

        # --- Decoder path với skip connections ---
        d3 = self.up3(b)                # (B, 64, H/4, W/4)
        skip3 = F.relu(self.bn_skip3(self.conv_skip3(x3)))
        d3 = torch.cat([d3, skip3], dim=1)   # (B, 96, H/4, W/4)
        d3 = F.relu(self.bn_cat3(d3))
        d3 = F.relu(self.conv_cat3(d3))       # (B, 32, H/4, W/4)
        d3 = self.decoder3(d3)
        d3 = self.dropout3(d3)

        d2 = self.up2(d3)               # (B, 32, H/2, W/2)
        skip2 = F.relu(self.bn_skip2(self.conv_skip2(x2)))
        d2 = torch.cat([d2, skip2], dim=1)   # (B, 48, H/2, W/2)
        d2 = F.relu(self.bn_cat2(d2))
        d2 = F.relu(self.conv_cat2(d2))       # (B, 16, H/2, W/2)
        d2 = self.decoder2(d2)
        d2 = self.dropout2(d2)

        d1 = self.up1(d2)               # (B, 16, H, W)
        skip1 = F.relu(self.bn_skip1(self.conv_skip1(x1)))
        d1 = torch.cat([d1, skip1], dim=1)   # (B, 24, H, W)
        d1 = F.relu(self.bn_cat1(d1))
        d1 = F.relu(self.conv_cat1(d1))       # (B, 8, H, W)
        d1 = self.decoder1(d1)
        d1 = self.dropout1(d1)

        # --- Output ---
        out = self.conv_out(d1)         # (B, n_classes, H, W)
        return out

# Kiểm tra số tham số
model = myModel(n_classes=3, dropout_prob=0.2)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")