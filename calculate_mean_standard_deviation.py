# calculate mean and std deviation

import cv2
import os
import numpy as np

image_dir = 'kaggle/input'
image_paths = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir)])

# Since the std can't be calculated by simply finding it for each image and averaging like
# the mean can be, to get the std we first calculate the overall mean in a first run then
# run it again to get the std.

mean = np.array([0., 0., 0.])
stdTemp = np.array([0., 0., 0.])
std = np.array([0., 0., 0.])

numSamples = len(image_paths)
print(numSamples)

for i in range(numSamples):
    image = cv2.imread(image_paths[i])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    im = image.astype(float) / 255.

    for j in range(3):
        mean[j] += np.mean(im[:, :, j])

mean = (mean / numSamples)

for i in range(numSamples):
    image = cv2.imread(image_paths[i])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    im = im.astype(float) / 255.

    for j in range(3):
        stdTemp[j] += ((im[:, :, j] - mean[j]) ** 2).sum() / (im.shape[0] * im.shape[1])

std = np.sqrt(stdTemp / numSamples)

print(mean)
print(std)

# out:
# [0.50707516 0.48654887 0.44091784]
# [0.26733429 0.25643846 0.27615047]

# import cv2
# import os
# import numpy as np
#
# # Đường dẫn đến thư mục chứa ảnh
# image_dir = 'kaggle/input'
# image_paths = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir)])
#
# # Đọc tất cả ảnh cùng lúc (giả sử ảnh có cùng kích thước)
# # Sử dụng list comprehension thay vì vòng for để tạo danh sách ảnh
# images = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in image_paths]
#
# # Chuyển danh sách ảnh thành ma trận lớn bằng np.stack
# all_images = np.stack(images).astype(np.float32) / 255.0
#
# # Tính mean trên tất cả ảnh theo các trục (0, 1, 2)
# mean = np.mean(all_images, axis=(0, 1, 2))
#
# # Tính std trên tất cả ảnh theo các trục (0, 1, 2)
# std = np.std(all_images, axis=(0, 1, 2))
#
# print("Mean:", mean)
# print("Std:", std)