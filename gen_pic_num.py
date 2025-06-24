import os
import cv2
import numpy as np
import random
from tqdm import tqdm

input_dir = 'single_pic/all'
output_dir = 'output_digits/aug_all'
os.makedirs(output_dir, exist_ok=True)

def random_augment(img):
    h, w = img.shape[:2]

    # 随机旋转 [-15°, 15°]
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h), borderValue=255)

    # 旋转矩阵
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=255)

    # 计算最大可用区域（不包含白边）
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    bound_w = int(w * abs_cos + h * abs_sin)
    bound_h = int(w * abs_sin + h * abs_cos)

    # 计算裁剪区域
    crop_size = int(min(h, w) * 0.8)
    x1 = (w - crop_size) // 2
    y1 = (h - crop_size) // 2
    img = rotated[y1:y1 + crop_size, x1:x1 + crop_size]

    # 随机裁剪或填充回原大小
    # img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    # 随机高斯模糊
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)


    return img

# 每张图片生成多少个增强版本
per_image = 2

img_list = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
counter = 0

for fname in tqdm(img_list):
    img = cv2.imread(os.path.join(input_dir, fname), cv2.IMREAD_GRAYSCALE)
    # 提取标签（假设是文件名中下划线后的数字，如 "djio_41.jpg" -> "41"）
    label = os.path.splitext(fname)[0].split('_')[-1]
    for i in range(per_image):
        aug = random_augment(img)
        out_path = os.path.join(output_dir, f"aug_{counter:04d}_{label}.png")
        cv2.imwrite(out_path, aug)
        counter += 1
        if counter >= 1000:
            break
    if counter >= 1000:
        break

print(f"增强完成，共生成 {counter} 张图片。")
