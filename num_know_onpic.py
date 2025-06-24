import os
import tqdm
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from num_model import SimpleCNN,ComplexCNN
from sklearn.metrics import classification_report, confusion_matrix
# 1. 定义一个简单的CNN模型（可以用训练好的模型替换）

# 2. 加载模型（在这里你可以加载训练好的模型）
def load_model():
    # model = SimpleCNN()
    model = ComplexCNN()

    model.load_state_dict(torch.load(f'model/Minist/{model.__class__.__name__}model.pth', map_location=torch.device('cpu'))) # 加载模型
    model.eval() #  设置模型为评估模式
    return model

# 3. 用训练好的模型预测单个数字图像
# 加载模型
model = load_model()

def predict_digit(img):
    def rotate_image(img, angle):
        h, w = img.shape # 获取图像的尺寸
        center = (w // 2, h // 2) #  计算图像中心点
        M = cv2.getRotationMatrix2D(center, angle, 1.0) # 计算旋转矩阵
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=0)  # 边缘填充为黑色 旋转图像
        return rotated

    def preprocess(img): #  预处理


        img = cv2.resize(img, (28, 28)) # 获取图像的尺寸
        img = img.astype('float32') / 255.0 #  归一化
        img = np.expand_dims(img, axis=-1)  # (28, 28, 1)   扩展维度
        img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # (1, 1, 28, 28)
        return img

    # 三种图像版本
    imgs = [
        preprocess(img), #  原始图像
        preprocess(rotate_image(img, 15)), #  旋转图像
        preprocess(rotate_image(img, -15)),
        preprocess(rotate_image(img, 10)),
        preprocess(rotate_image(img, -10)),
        preprocess(rotate_image(img, 5)),
        preprocess(rotate_image(img, -5))
    ]


    # 存储预测结果
    best_pred = None # 存储预测结果
    best_conf = -1 # 最高概率

    with torch.no_grad(): #  不记录梯度
        for i, im in enumerate(imgs):
            # 数据预处理
            transform = transforms.Compose([
                transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值与标准差
            ])
            im = transform(im)  # shape: [1, 28, 28]

            output = model(im) #  得到输出
            # print(output)
            prob = F.softmax(output, dim=1) #  得到概率

            conf, pred = torch.max(prob, 1) #  得到概率和预测结果
            if conf.item() > best_conf:
                best_conf = conf.item()
                best_pred = pred.item()

    return best_pred

def edge_detection_and_segmentation(image_path): #边缘检测
    # 读取图像
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 将图像转换为灰度图像

    # 预处理：高斯模糊（去噪）
    gray = cv2.GaussianBlur(gray, (5, 5), 0) #  高斯模糊 对灰度图应用高斯模糊，降低图像中的细节噪声，提高边缘检测的鲁棒性

    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 形态学操作：膨胀连接数字片段，然后轻度腐蚀防止过度粘连
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 对轮廓按从左到右排序（根据x坐标）
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    return contours, img, gray
# 5. 识别数字的数量（判断是一位数还是两位数）
def detect_digit_count(contours):
    # 根据轮廓数量来判断数字的个数
    if len(contours) == 1:
        return 1  # 一位数
    else:
        return 2  # 两位数
def normalize_background(img):
    """
    将灰色底统一处理成纯黑背景。
    适用于 MNIST 风格的“黑底白字”图像。

    参数：
        img: 输入灰度图像（0~255）

    返回：
        处理后的图像
    """
    # 如果背景是亮色（白底黑字），先反转
    mean_val = np.mean(img)

    # 阈值以下的像素归为0（黑色）
    img[img < mean_val] = 0
    img[img >= mean_val] = 255

    black_ratio = np.sum(img ==0) / img.size  # 统计接近黑的像素比例
    if black_ratio < 0.5:  # 表示整体偏白
        img = 255 - img
        mean_val=255-mean_val
    return img

def should_thicken(img, white_thresh=100):
    """
    判断图像中的白色像素是否足够少（笔画细），若是则返回 True。
    """
    white_pixels = cv2.countNonZero(img)
    return white_pixels < white_thresh
def thicken_strokes(img, kernel_size=2, iterations=1):
    """
    使用膨胀操作加粗图像中的数字笔画。

    参数：
        img: 输入灰度图像（黑底白字）
        kernel_size: 卷积核大小（越大越粗）
        iterations: 操作次数（次数越多越粗）

    返回：
        加粗后的图像
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    thickened = cv2.dilate(img, kernel, iterations=iterations)
    return thickened
# 6. 识别完整图片中的数字
def recognize_image(image_path):
    contours, img, gray = edge_detection_and_segmentation(image_path)

    digit_imgs = []
    bounding_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 100:
            digit_img = gray[y:y + h, x:x + w]
            digit_imgs.append((x, digit_img))
            bounding_boxes.append((x, y, w, h))

    if not digit_imgs:
        raise ValueError('没有找到任何数字！')

    digit_imgs.sort(key=lambda x: x[0])
    bounding_boxes.sort(key=lambda x: x[0])

    result = ""

    for i, (x, digit_img) in enumerate(digit_imgs):
        try:
            # 图像预处理
            digit_img = cv2.GaussianBlur(digit_img, (3, 3), 0)
            h, w = digit_img.shape
            scale = 24 / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            digit_img_resized = cv2.resize(digit_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            digit_img_resized = normalize_background(digit_img_resized)

            top = (28 - new_h) // 2
            bottom = 28 - new_h - top
            left = (28 - new_w) // 2
            right = 28 - new_w - left

            digit_img_padded = cv2.copyMakeBorder(
                digit_img_resized,
                top=top, bottom=bottom, left=left, right=right,
                borderType=cv2.BORDER_CONSTANT,
                value=0
            )

            if should_thicken(digit_img_padded, white_thresh=100):
                # cv2.imshow("Digit", digit_img_padded)
                # cv2.waitKey(0)  # 等待按键
                # cv2.destroyAllWindows()

                digit_img_padded = thicken_strokes(digit_img_padded, kernel_size=2, iterations=1) #  粗化
                #
                # cv2.imshow("Digit", digit_img_padded)
                # cv2.waitKey(0)  # 等待按键
                # cv2.destroyAllWindows()

            pred = predict_digit(digit_img_padded)
            result += str(pred)

            # 获取框的位置并画框 + 文字
            x0, y0, w0, h0 = bounding_boxes[i]
            cv2.rectangle(img, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 2)  # 绿色框
            cv2.putText(img, str(pred), (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # 红色数字`
        except:
            pass

    # 显示最终结果图像
    cv2.imshow("Prediction Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 也可选择保存图像
    cv2.imwrite("plot/number/test/prediction_result.jpg", img)

    return int(result)





# 7. 主程序
if __name__ == "__main__":
    # 图片路径
    image_path = "plot/number/test/white.png"
    # image_path = "plot/number/test/black.png"
    # image_path = "plot/number/test/handwrite.png"


    # 识别
    try:
        number = recognize_image(image_path)
        print("识别结果：", number)

    except Exception as e:
        print("识别失败：", e)
