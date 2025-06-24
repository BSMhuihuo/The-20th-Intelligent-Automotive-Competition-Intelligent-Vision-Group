import cv2
import numpy as np
import os
import random
import csv

# 创建输出目录
# os.makedirs("output_digits_black", exist_ok=True)
white=False
# 加载图像并转为灰度
image = cv2.imread("single_pic/digit_black.png")
# image=cv2.imread("single_pic/digit_white.jpg")
line_threshold = 50
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 边缘检测（增强准确性）
edges = cv2.Canny(gray, 100, 200)
kernel = np.ones((3, 3), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)
edges = cv2.erode(edges, kernel, iterations=1)

# 提取轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bounding_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 50]

# 按垂直位置粗分成三行
bounding_boxes.sort(key=lambda b: b[1])
rows = [[], [], []]


for box in bounding_boxes:
    x, y, w, h = box
    cy = y + h // 2
    if not rows[0] or abs(cy - np.mean([r[1] + r[3] // 2 for r in rows[0]])) < line_threshold:
        rows[0].append(box)
    elif not rows[1] or abs(cy - np.mean([r[1] + r[3] // 2 for r in rows[1]])) < line_threshold:
        rows[1].append(box)
    else:
        rows[2].append(box)
# 每行按水平位置排序
for row in rows:
    row.sort(key=lambda b: b[0])

# 提取原始高分辨率数字图像（灰度）
digits = []
for row in rows:
    for x, y, w, h in row:
        digit = gray[y:y+h, x:x+w]
        digits.append(digit)

# 确保提取了30个数字
assert len(digits) == 30, f"应为30个数字，当前为 {len(digits)}"


if white==True:
    # 生成两位数图像（高清合成，白色间隔）
    labels = []
    for idx in range(1000):
        d1 = random.randint(0, 9)
        d2 = random.randint(0, 9)
        digit1 = random.choice([digits[d1], digits[d1 + 10], digits[d1 + 20]])
        digit2 = random.choice([digits[d2], digits[d2 + 10], digits[d2 + 20]])




        h1, w1 = digit1.shape
        h2, w2 = digit2.shape
        spacing = 10  # 白色空隙宽度

        height = max(h1, h2)
        width = w1 + spacing + w2

        canvas = np.ones((height, width), dtype=np.uint8) * 255  # 白底

        # 垂直居中粘贴
        canvas[(height - h1) // 2:(height - h1) // 2 + h1, 0:w1] = digit1
        canvas[(height - h2) // 2:(height - h2) // 2 + h2, w1 + spacing:] = digit2

        # === 添加随机白边（偏移位置） ========
        top = random.randint(5, 100)
        bottom = random.randint(5, 100)
        left = random.randint(5, 100)
        right = random.randint(5, 100)

        canvas_padded = cv2.copyMakeBorder(
            canvas,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv2.BORDER_CONSTANT,
            value=255  # 白色边框
        )
        # ==== 加入旋转 ====
        angle = random.uniform(-15, 15)  # 可调范围
        (h, w) = canvas_padded.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        canvas_rotated = cv2.warpAffine(canvas_padded, M, (w, h), borderValue=255)
        # ==================
        # =================================

        filename = f"output_digits/output_digits_white/white_img_{idx:04d}.png"
        cv2.imwrite(filename, canvas_padded)
        labels.append((filename, d1 * 10 + d2))

    # 写入标签
    with open("output_digits/output_digits_white/white_labels.csv", mode="w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(labels)
    print("已完成：生成 1000 张高清图像，白色背景、白色间隔，保存在 output_digits/")

else:
    # 生成两位数图像（高清合成，白色间隔）
    labels = []
    for idx in range(1000):
        d1 = random.randint(0, 9)
        d2 = random.randint(0, 9)
        digit1 = random.choice([digits[d1], digits[d1 + 10], digits[d1 + 20]])
        digit2 = random.choice([digits[d2], digits[d2 + 10], digits[d2 + 20]])

        h1, w1 = digit1.shape
        h2, w2 = digit2.shape
        spacing = 10  # 空隙宽度

        height = max(h1, h2)
        width = w1 + spacing + w2

        canvas = np.ones((height, width), dtype=np.uint8) * 34  # 白底

        # 垂直居中粘贴
        canvas[(height - h1) // 2:(height - h1) // 2 + h1, 0:w1] = digit1
        canvas[(height - h2) // 2:(height - h2) // 2 + h2, w1 + spacing:] = digit2


        # === 添加随机白边（偏移位置） ========
        top = random.randint(5, 100)
        bottom = random.randint(5, 100)
        left = random.randint(5, 100)
        right = random.randint(5, 100)

        canvas_padded = cv2.copyMakeBorder(
            canvas,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv2.BORDER_CONSTANT,
            value=34  # 白色边框
        )
        # ==== 加入旋转 ====
        angle = random.uniform(-15, 15)  # 可调范围
        (h, w) = canvas_padded.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        canvas_rotated = cv2.warpAffine(canvas_padded, M, (w, h), borderValue=34)
        # ==================
        # =================================

        filename = f"output_digits/output_digits_black/black_img_{idx:04d}.png"
        cv2.imwrite(filename, canvas_padded)
        labels.append((filename, d1 * 10 + d2))

    # 写入标签
    with open("output_digits/output_digits_black/black_labels.csv", mode="w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(labels)
    print("已完成：生成 1000 张高清图像，黑色背景、黑色间隔，保存在 output_digits_black/")



