from torch.utils.data import DataLoader
from dataset import SemanticSegmentationDataset, MyDataset
from torchvision import transforms
import torch
from torch.optim import Adam
from learner import evaluate
from model_10 import myModel
import torch.nn as nn
from utils import count_parameters
import random
import matplotlib.pyplot as plt
import numpy as np
import pickle

dataset = SemanticSegmentationDataset(
    image_dir='kaggle/input',
    label_dir='kaggle/label',
)

with open('mean_std.pkl', 'rb') as f:
    data = pickle.load(f)
    mean = data['mean']
    std = data['std']

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=mean, std=std)
])

val_dataset = MyDataset(dataset, transform=val_transform)

val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_classes = len(dataset.class_colors)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = myModel(num_classes).to(device)

count_parameters(model) # Đếm số parameters của model

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
num_epochs = 5

# Hàm chuyển mask (nhãn) thành ảnh RGB
def mask_to_rgb(mask, class_colors):
    """Chuyển nhãn (label mask) thành ảnh RGB dựa trên class_colors."""
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for rgb, idx in class_colors.items():
        rgb_mask[mask == idx] = rgb  # Gán màu theo class

    return rgb_mask


# Hàm lấy ngẫu nhiên một số mẫu từ bộ dữ liệu
def get_random_samples(dataloader, num_samples):
    """Lấy ngẫu nhiên một số ảnh từ bộ dữ liệu validation."""
    indices = random.sample(range(len(dataloader.dataset)), num_samples)  # Chọn ngẫu nhiên các chỉ số ảnh
    images_list, labels_list = [], []

    for idx in indices:
        image, label = dataloader.dataset[idx]  # Lấy ảnh và nhãn từ dataset theo chỉ số ngẫu nhiên
        images_list.append(image)
        labels_list.append(label)

    return torch.stack(images_list), torch.stack(labels_list)


# Hàm hiển thị ảnh, nhãn thực tế và nhãn dự đoán
def visualize_predictions(images, labels, preds, class_colors, num_samples=3):
    """Hiển thị ảnh đầu vào, nhãn thực tế và nhãn dự đoán."""
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))

    for j in range(num_samples):
        img = images[j].cpu().numpy().transpose(1, 2, 0)  # Chuyển Tensor về numpy
        lbl = labels[j].cpu().numpy()
        pred = preds[j].cpu().numpy()

        # Chuyển mask sang ảnh màu
        lbl_rgb = mask_to_rgb(lbl, class_colors)
        pred_rgb = mask_to_rgb(pred, class_colors)

        axes[j, 0].imshow(img)
        axes[j, 0].set_title("Input Image")
        axes[j, 0].axis("off")

        axes[j, 1].imshow(lbl_rgb)
        axes[j, 1].set_title("Ground Truth")
        axes[j, 1].axis("off")

        axes[j, 2].imshow(pred_rgb)
        axes[j, 2].set_title("Predicted Mask")
        axes[j, 2].axis("off")

    plt.show()


# Kiểm tra và hiển thị dự đoán
model = torch.jit.load("checkpoints/22139078_22139044_best_model.pt")

# Đánh giá mô hình
model.eval()
with torch.no_grad():
    images, labels = get_random_samples(val_dataloader, 50)


    images = images.to(device)
    labels = labels.to(device)

    # Dự đoán kết quả
    preds = model(images).argmax(dim=1)  # Lấy class có giá trị lớn nhất

    # Hiển thị kết quả
    visualize_predictions(images, labels, preds, dataset.class_colors, num_samples=50)

epoch_loss_val, mAcc_val, mIoU_val = evaluate(model, val_dataloader, criterion, device, num_classes)
print(f"Validation Loss: {epoch_loss_val:.4f}, Mean Accuracy: {mAcc_val:.4f}, Mean IoU: {mIoU_val:.4f}")
