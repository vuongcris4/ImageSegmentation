import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import Adam
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
import copy

class SemanticSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir)])
        self.label_paths = sorted([os.path.join(label_dir, lbl) for lbl in os.listdir(label_dir)])
        self.class_colors = {
            (2, 0, 0): 0,       
            (127, 0, 0): 1,     
            (248, 163, 191): 2  
        }
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(self.label_paths[idx])
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        label_mask = np.zeros(label.shape[:2], dtype=np.uint8)
        for rgb, idx in self.class_colors.items():
            label_mask[np.all(label == rgb, axis=-1)] = idx

        if self.transform:
            image = self.transform(image)
            label_mask = torch.from_numpy(label_mask).long()

        return image, label_mask

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

dataset = SemanticSegmentationDataset(
    image_dir='kaggle/input',
    label_dir='kaggle/label',
    transform=train_transform)


def train_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss = 0.0  
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    pbar = tqdm(dataloader, desc='Training', unit='batch')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)       
        optimizer.zero_grad()
        outputs = model(images)    
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()   
        running_loss += loss.item() * images.size(0)      
        preds = torch.argmax(outputs, dim=1)     
        accuracy_metric(preds, labels)
        iou_metric(preds, labels)
        pbar.set_postfix({
            'Batch Loss': f'{loss.item():.4f}',
            'Mean Accuracy': f'{accuracy_metric.compute():.4f}',
            'Mean IoU': f'{iou_metric.compute():.4f}',
        }) 
    epoch_loss = running_loss / len(dataloader.dataset)  
    mean_accuracy = accuracy_metric.compute().cpu().numpy()
    mean_iou = iou_metric.compute().cpu().numpy()
   
    return epoch_loss, mean_accuracy, mean_iou

def evaluate(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0    
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    pbar = tqdm(dataloader, desc='Evaluating', unit='batch')
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            # Update metrics
            accuracy_metric(preds, labels)
            iou_metric(preds, labels)
            # Update tqdm description with metrics
            pbar.set_postfix({
                'Batch Loss': f'{loss.item():.4f}',
                'Mean Accuracy': f'{accuracy_metric.compute():.4f}',
                'Mean IoU': f'{iou_metric.compute():.4f}',
            })
    
    epoch_loss = running_loss / len(dataloader.dataset)
    mean_accuracy = accuracy_metric.compute().cpu().numpy()
    mean_iou = iou_metric.compute().cpu().numpy()
    
    return epoch_loss, mean_accuracy, mean_iou
 
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

total_size = len(dataset)
train_size = int(0.8 * total_size)  
val_size = total_size - train_size  
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
classes = 3  
model = myModel(classes)
def count_parameters(model):  
    return sum(p.numel() for p in model.parameters())
total_params = count_parameters(model)
print(f"Total parameters: {total_params}")
model.to(device)
model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()  
optimizer = Adam(model.parameters(), lr=0.001)
num_epochs = 40

epoch_saved = 0
best_val_mAcc = 0.0  
best_model_state = None

for epoch in range(num_epochs):
    epoch_loss_train, mAcc_train, mIoU_train = train_epoch(model, train_dataloader, criterion, optimizer, device, classes)
    epoch_loss_val, mAcc_val, mIoU_val = evaluate(model, val_dataloader, criterion, device, classes)
    
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {epoch_loss_train:.4f}, Mean Accuracy: {mAcc_train:.4f}, Mean IoU: {mIoU_train:.4f}")
    print(f"Validation Loss: {epoch_loss_val:.4f}, Mean Accuracy: {mAcc_val:.4f}, Mean IoU: {mIoU_val:.4f}")

    if mAcc_val >= best_val_mAcc:
        epoch_saved = epoch + 1 
        best_val_mAcc = mAcc_val
        best_model_state = copy.deepcopy(model.state_dict())
    
print("===================")
print(f"Best Model at epoch : {epoch_saved}")
model.load_state_dict(best_model_state)
if isinstance(model, torch.nn.DataParallel):
    model = model.module
model_save = torch.jit.script(model)
model_save.save("18.pt")
# Check again
model = torch.jit.load("18.pt")
epoch_loss_val, mAcc_val, mIoU_val = evaluate(model, val_dataloader, criterion, device, classes)
print(f"Validation Loss: {epoch_loss_val:.4f}, Mean Accuracy: {mAcc_val:.4f}, Mean IoU: {mIoU_val:.4f}")