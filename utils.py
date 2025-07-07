import pickle

import cv2
import numpy as np
from torch.utils.data import random_split
from tqdm import tqdm


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

def train_val_split(dataset):
    # 80% train, 20% test
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    return train_dataset, val_dataset

def calculate_mean_standard(dataset):
    mean = np.array([0., 0., 0.])
    stdTemp = np.array([0., 0., 0.])
    std = np.array([0., 0., 0.])

    numSamples = len(dataset)
    pbar = tqdm(range(numSamples), desc="calculating mean")
    for i in pbar:
        image, label = dataset[i]

        im = image.astype(float) / 255.

        for j in range(3):
            mean[j] += np.mean(im[:, :, j])

    mean = (mean / numSamples)

    pbar = tqdm(range(numSamples), desc="calculating std")
    for i in pbar:
        image, label = dataset[i]

        im = image.astype(float) / 255.

        for j in range(3):
            stdTemp[j] += ((im[:, :, j] - mean[j]) ** 2).sum() / (im.shape[0] * im.shape[1])

    std = np.sqrt(stdTemp / numSamples)

    with open("mean_std.pkl", "wb") as f:
        pickle.dump({'mean': mean, 'std': std}, f)
    
    print("mean: ", mean)
    print("std: ", std)

    return mean, std