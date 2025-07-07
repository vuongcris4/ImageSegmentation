import torch
import torch.nn as nn
import torch.nn.functional as F

# Định nghĩa ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2, stride=1)
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

# Định nghĩa AttentionGate
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        return x * psi

# Định nghĩa mô hình myModel với Attention
class myModel(nn.Module):
    def __init__(self, n_classes, dropout_prob=0.2):
        super(myModel, self).__init__()
        
        # --- Encoder ---
        self.encoder1 = ResidualBlock(3, 8)    
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ResidualBlock(8, 16)   
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ResidualBlock(16, 32)  
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ResidualBlock(32, 64)  

        # --- Attention Gates ---
        self.ag3 = AttentionGate(F_g=64, F_l=32, F_int=8)
        self.ag2 = AttentionGate(F_g=32, F_l=16, F_int=8)
        self.ag1 = AttentionGate(F_g=16, F_l=8, F_int=8)

        # --- Skip connections ---
        self.conv_skip3 = nn.Conv2d(32, 64, kernel_size=1)  
        self.bn_skip3 = nn.BatchNorm2d(64)
        self.conv_skip2 = nn.Conv2d(16, 32, kernel_size=1)  
        self.bn_skip2 = nn.BatchNorm2d(32)
        self.conv_skip1 = nn.Conv2d(8, 16, kernel_size=1)   
        self.bn_skip1 = nn.BatchNorm2d(16)

        # --- Dropout ---
        self.dropout_bottleneck = nn.Dropout2d(p=dropout_prob)

        # --- Decoder ---
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn_add3 = nn.BatchNorm2d(64)  
        self.conv_reduce3 = nn.Conv2d(64, 32, kernel_size=1)  
        self.decoder3 = ResidualBlock(32, 32)
        self.dropout3 = nn.Dropout2d(p=dropout_prob)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn_add2 = nn.BatchNorm2d(32)  
        self.conv_reduce2 = nn.Conv2d(32, 16, kernel_size=1)  
        self.decoder2 = ResidualBlock(16, 16)
        self.dropout2 = nn.Dropout2d(p=dropout_prob)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn_add1 = nn.BatchNorm2d(16)  
        self.conv_reduce1 = nn.Conv2d(16, 8, kernel_size=1)   
        self.decoder1 = ResidualBlock(8, 8)
        self.dropout1 = nn.Dropout2d(p=dropout_prob)

        # --- Output layer ---
        self.conv_out = nn.Conv2d(8, n_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        x1 = self.encoder1(x)           
        p1 = self.pool1(x1)             
        x2 = self.encoder2(p1)          
        p2 = self.pool2(x2)             
        x3 = self.encoder3(p2)          
        p3 = self.pool3(x3)             
        b = self.bottleneck(p3)         
        b = self.dropout_bottleneck(b)  
        
        # --- Decoder với Attention và skip connections ---
        d3 = self.up3(b)                
        skip3 = self.ag3(d3, x3)  # Áp dụng AttentionGate
        skip3 = F.relu(self.bn_skip3(self.conv_skip3(skip3)))  
        d3 = d3 + skip3                 
        d3 = F.relu(self.bn_add3(d3))   
        d3 = F.relu(self.conv_reduce3(d3))  
        d3 = self.decoder3(d3)          
        d3 = self.dropout3(d3)

        d2 = self.up2(d3)               
        skip2 = self.ag2(d2, x2)  # Áp dụng AttentionGate
        skip2 = F.relu(self.bn_skip2(self.conv_skip2(skip2)))  
        d2 = d2 + skip2                 
        d2 = F.relu(self.bn_add2(d2))   
        d2 = F.relu(self.conv_reduce2(d2))  
        d2 = self.decoder2(d2)          
        d2 = self.dropout2(d2)

        d1 = self.up1(d2)               
        skip1 = self.ag1(d1, x1)  # Áp dụng AttentionGate
        skip1 = F.relu(self.bn_skip1(self.conv_skip1(skip1)))  
        d1 = d1 + skip1                 
        d1 = F.relu(self.bn_add1(d1))   
        d1 = F.relu(self.conv_reduce1(d1))  
        d1 = self.decoder1(d1)          
        d1 = self.dropout1(d1)

        # --- Output ---
        out = self.conv_out(d1)         
        return out

# Kiểm tra số tham số
model = myModel(n_classes=3, dropout_prob=0.2)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Tổng số tham số huấn luyện (với Attention): {total_params:,}")