import torch
import torch.nn as nn
import torch.nn.functional as F

# Định nghĩa ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
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

# Định nghĩa mô hình myModel đã sửa đổi
class myModel(nn.Module):
    def __init__(self, n_classes, dropout_prob=0.2):
        super(myModel, self).__init__()
        
        # --- Encoder ---
        self.encoder1 = ResidualBlock(3, 8)    # Từ 3 kênh đầu vào -> 8 kênh
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ResidualBlock(8, 16)   # 8 -> 16 kênh
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ResidualBlock(16, 32)  # 16 -> 32 kênh
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ResidualBlock(32, 64)  # 32 -> 64 kênh

        # --- Skip connections với Conv1x1, BatchNorm, ReLU ---
        self.conv_skip1 = nn.Conv2d(8, 8, kernel_size=1)
        self.bn_skip1 = nn.BatchNorm2d(8)
        self.conv_skip2 = nn.Conv2d(16, 16, kernel_size=1)
        self.bn_skip2 = nn.BatchNorm2d(16)
        self.conv_skip3 = nn.Conv2d(32, 32, kernel_size=1)
        self.bn_skip3 = nn.BatchNorm2d(32)

        # --- Dropout sau Bottleneck ---
        self.dropout_bottleneck = nn.Dropout2d(p=dropout_prob)

        # --- Decoder ---
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn_cat3 = nn.BatchNorm2d(64 + 32)  # 64 (từ bottleneck) + 32 (từ skip)
        self.decoder3 = ResidualBlock(64 + 32, 32)  # Giảm từ 96 -> 32 kênh
        self.dropout3 = nn.Dropout2d(p=dropout_prob)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn_cat2 = nn.BatchNorm2d(32 + 16)  # 32 (từ decoder3) + 16 (từ skip)
        self.decoder2 = ResidualBlock(32 + 16, 16)  # Giảm từ 48 -> 16 kênh
        self.dropout2 = nn.Dropout2d(p=dropout_prob)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn_cat1 = nn.BatchNorm2d(16 + 8)   # 16 (từ decoder2) + 8 (từ skip)
        self.decoder1 = ResidualBlock(16 + 8, 8)    # Giảm từ 24 -> 8 kênh
        self.dropout1 = nn.Dropout2d(p=dropout_prob)

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
        b = self.dropout_bottleneck(b)  # Áp dụng dropout

        # --- Skip connections với Conv1x1, BatchNorm, ReLU ---
        skip1 = F.relu(self.bn_skip1(self.conv_skip1(x1)))  # (B, 8, H, W)
        skip2 = F.relu(self.bn_skip2(self.conv_skip2(x2)))  # (B, 16, H/2, W/2)
        skip3 = F.relu(self.bn_skip3(self.conv_skip3(x3)))  # (B, 32, H/4, W/4)

        # --- Decoder path với skip connections ---
        # Giai đoạn 3
        d3 = self.up3(b)                # (B, 64, H/4, W/4)
        d3 = torch.cat([d3, skip3], dim=1) # (B, 64 + 32, H/4, W/4)
        d3 = F.relu(self.bn_cat3(d3))   # Áp dụng BatchNorm và ReLU
        d3 = self.decoder3(d3)          # Giảm từ 96 -> 32 kênh
        d3 = self.dropout3(d3)

        # Giai đoạn 2
        d2 = self.up2(d3)               # (B, 32, H/2, W/2)
        d2 = torch.cat([d2, skip2], dim=1) # (B, 32 + 16, H/2, W/2)
        d2 = F.relu(self.bn_cat2(d2))   # Áp dụng BatchNorm và ReLU
        d2 = self.decoder2(d2)          # Giảm từ 48 -> 16 kênh
        d2 = self.dropout2(d2)

        # Giai đoạn 1
        d1 = self.up1(d2)               # (B, 16, H, W)
        d1 = torch.cat([d1, skip1], dim=1) # (B, 16 + 8, H, W)
        d1 = F.relu(self.bn_cat1(d1))   # Áp dụng BatchNorm và ReLU
        d1 = self.decoder1(d1)          # Giảm từ 24 -> 8 kênh
        d1 = self.dropout1(d1)

        # --- Output ---
        out = self.conv_out(d1)         # (B, n_classes, H, W)
        return out

# --- Kiểm tra số tham số ---
model = myModel(n_classes=3, dropout_prob=0.2)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters (with Dropout): {total_params:,}")