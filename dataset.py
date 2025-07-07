from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torch

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
            (248, 163, 191): 2  # Màu (R=248, G=163, B=191) tương ứng với lớp 2
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #         # Chuyển đổi không gian màu của ảnh từ BGR (mặc định của OpenCV) sang RGB
        # print("image", image, "len", (image.shape))

        label = cv2.imread(self.label_paths[idx])
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        # print("image", label, "len", (label.shape))


        # Tạo một mảng NumPy mới chứa toàn số 0, có cùng kích thước chiều cao và chiều rộng với ảnh nhãn
        # Kiểu dữ liệu là uint8 (số nguyên không dấu 8 bit), phù hợp để lưu chỉ số lớp (0-255)
        label_mask = np.zeros(label.shape[:2], dtype=np.uint8)
        # Lặp qua từng cặp (màu RGB, chỉ số lớp) trong dictionary class_colors
        for rgb, idx in self.class_colors.items():
            label_mask[np.all(label == rgb, axis=-1)] = idx

        # if self.transform:
        #     image = self.transform(image)
        label_mask = torch.from_numpy(label_mask).long()

        return image, label_mask


class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
