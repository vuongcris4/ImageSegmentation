from torch.utils.data import DataLoader
from dataset import SemanticSegmentationDataset, MyDataset
from torchvision import transforms
import torch
from torch.optim import Adam
from learner import Learner, evaluate
from model_10 import myModel
import torch.nn as nn
from utils import count_parameters, train_val_split

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

dataset = SemanticSegmentationDataset(
    image_dir='kaggle/input',
    label_dir='kaggle/label',
)

train_dataset, val_dataset = train_val_split(dataset)

train_dataset = MyDataset(train_dataset, train_transform)
val_dataset = MyDataset(val_dataset, val_transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_classes = len(dataset.class_colors)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = myModel(num_classes).to(device)

count_parameters(model) # Đếm số parameters của model

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
num_epochs = 5

# GIẢ SỬ MẤT ĐIỆN, TRAIN LẠI
model3 = myModel(num_classes).to(device)
checkpoint = torch.load("checkpoints/22139078_22139044_last_model.pt", weights_only=False)
optimizer = Adam(model.parameters(), lr=0.001)
optimizer.load_state_dict(checkpoint["optimizer"])
learner = Learner(model3, criterion, optimizer, device, num_epochs)
learner.fit(train_dataloader, val_dataloader, 3, start_epoch=checkpoint["epoch"]) # train và lưu best model
epoch_loss_val, mAcc_val, mIoU_val = evaluate(model3, val_dataloader, criterion, device, num_classes)
print(f"Validation Loss: {epoch_loss_val:.4f}, Mean Accuracy: {mAcc_val:.4f}, Mean IoU: {mIoU_val:.4f}")