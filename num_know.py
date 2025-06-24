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

    # cv2.imshow("Digit", img)
    # cv2.waitKey(0)  # 等待按键
    # cv2.destroyAllWindows()
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
    # print("开始边缘检测")
    # 读取图像
    img = cv2.imread(image_path)

    # cv2.imshow("Digit", img)
    # cv2.waitKey(0)  # 等待按键
    # cv2.destroyAllWindows()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 将图像转换为灰度图像

    # cv2.imshow("Digit", gray)
    # cv2.waitKey(0)  # 等待按键
    # cv2.destroyAllWindows()

    # 预处理：高斯模糊（可选，去噪）
    gray = cv2.GaussianBlur(gray, (5, 5), 0) #  高斯模糊 对灰度图应用高斯模糊，降低图像中的细节噪声，提高边缘检测的鲁棒性

    # cv2.imshow("Digit", gray)
    # cv2.waitKey(0)  # 等待按键
    # cv2.destroyAllWindows()

    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # cv2.imshow("Digit", edges)
    # cv2.waitKey(0)  # 等待按键
    # cv2.destroyAllWindows()

    # 形态学操作：膨胀连接数字片段，然后轻度腐蚀防止过度粘连
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # cv2.imshow("Digit", edges)
    # cv2.waitKey(0)  # 等待按键
    # cv2.destroyAllWindows()

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 对轮廓按从左到右排序（根据x坐标）
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # print("边缘检测完成")

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
    # print(f"背景色占比：{black_ratio:.2%}")
    # if mean_val > 255/2:  # 表示整体偏白
    #     img = 255 - img
    #     mean_val=255-mean_val
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


    # 判断数字数量
    # digit_count = detect_digit_count(contours)

    # print(f"检测到 {len(contours)} 位数字")

    # 用于保存数字区域
    digit_imgs = []

    # 根据轮廓提取数字区域并进行处理
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 100:  # 过滤噪声，确保数字较大
            digit_img = gray[y:y + h, x:x + w]
            digit_imgs.append((x, digit_img))

    if not digit_imgs:
        raise ValueError('没有找到任何数字！')

    # 左右排序，确保数字顺序正确
    digit_imgs.sort(key=lambda x: x[0])

    # 识别每一块数字
    result = ""

    for x, digit_img in digit_imgs:
        # cv2.imshow("Digit", digit_img)
        # cv2.waitKey(0)  # 等待按键
        # cv2.destroyAllWindows()


        # 1. 等比缩小前轻微模糊
        digit_img = cv2.GaussianBlur(digit_img, (3, 3), 0)

        # 2. 后续步骤同前
        h, w = digit_img.shape
        scale = 24 / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        digit_img_resized = cv2.resize(digit_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # cv2.imshow("Digit", digit_img_resized)
        # cv2.waitKey(0)  # 等待按键
        # cv2.destroyAllWindows()

        digit_img_resized= normalize_background(digit_img_resized)

        # cv2.imshow("Digit", digit_img_resized)
        # cv2.waitKey(0)  # 等待按键
        # cv2.destroyAllWindows()

        # 3. 在28x28画布中居中填充
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

        # 4. 判断是否需要加粗
        if should_thicken(digit_img_padded, white_thresh=100):
            digit_img_padded = thicken_strokes(digit_img_padded, kernel_size=2, iterations=1)
            # digit_img_padded[digit_img_padded>95]=255

        # cv2.imshow("Digit", digit_img_padded)
        # cv2.waitKey(0)  # 等待按键
        # cv2.destroyAllWindows()
        # 5. 预测
        pred = predict_digit(digit_img_padded)
        result += str(pred)

    return int(result)


def enhanced_predict_digits(img_batch):
    """
    对一个 batch 的图像做多角度预测，返回每个图像的最佳预测值。

    参数：
        img_batch: numpy array, shape = (N, 28, 28)
        model: pytorch 模型

    返回：
        numpy array, shape = (N,)，每个图像的最佳预测标签
    """
    angles = [0, -5, 5, -10, 10, -15, 15]
    all_preds = []


    # cv2.imshow("Digit", img_batch[2])
    # cv2.waitKey(0)  # 等待按键
    # cv2.destroyAllWindows()

    transform = transforms.Normalize((0.1307,), (0.3081,))

    img_batch = img_batch.astype(np.float32)
    batch_size = img_batch.shape[0]

    for angle in angles:
        rotated_batch = []
        for i in range(batch_size):
            img = img_batch[i]

            h, w = img.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), borderValue=0)
            rotated_batch.append(rotated)


        rotated_batch = np.stack(rotated_batch, axis=0)  # (N, 28, 28)
        rotated_batch = torch.tensor(rotated_batch).unsqueeze(1)  # (N, 1, 28, 28)

        rotated_batch = transform(rotated_batch)



        with torch.no_grad():
            logits = model(rotated_batch)
            probs = F.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)  # 每个图像的置信度和预测结果

            all_preds.append((confs, preds))


    # 每个图像选择置信度最高的角度对应的预测值
    final_preds = []
    for i in range(batch_size):
        best_conf = -1
        best_pred = -1
        for confs, preds in all_preds:
            if confs[i].item() > best_conf:
                best_conf = confs[i].item()
                best_pred = preds[i].item()
        final_preds.append(best_pred)

    return np.array(final_preds)
def recognize_images_batch(folder_path):
    """
    批量识别多张图片中的数字（按 batch 推理）

    参数：
        image_paths: List[str]，图片路径列表
        model: 已加载的模型（如一个 PyTorch 模型）

    返回：
        Dict[str, int]，键为图片路径，值为识别出的整数数字
    """
    all_digit_imgs = []
    digit_info = []  # [(image_path, index_in_image)]

    results = {}

    # 第一步：预处理，提取所有数字图像
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            try:
                contours, img, gray = edge_detection_and_segmentation(image_path)
                print(f"[{image_path}] 检测到 {len(contours)} 个数字")

                digit_imgs = []

                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w * h > 100:
                        digit_img = gray[y:y + h, x:x + w]
                        digit_imgs.append((x, digit_img))

                if not digit_imgs:
                    raise ValueError('没有找到任何数字！')

                digit_imgs.sort(key=lambda x: x[0])

                for i, (x, digit_img) in enumerate(digit_imgs):
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
                        digit_img_padded = thicken_strokes(digit_img_padded, kernel_size=2, iterations=1)

                    digit_img_padded = digit_img_padded.astype(np.float32) / 255.0  # 归一化
                    all_digit_imgs.append(digit_img_padded)
                    digit_info.append((image_path, i))

            except Exception as e:
                print(f"处理 {image_path} 时出错: {e}")
                results[image_path] = None

    if not all_digit_imgs:
        return results

    # 第二步：打包成 batch，模型推理
    input_batch = np.stack(all_digit_imgs, axis=0)  # (N, 28, 28)
    input_batch = np.expand_dims(input_batch, axis=1)  # (N, 1, 28, 28)

    import torch
    # input_tensor = torch.tensor(input_batch)

    preds = enhanced_predict_digits(input_batch[:, 0, :, :])



    # 第三步：还原回各张图片
    image_results = {}
    for (image_path, _), pred in zip(digit_info, preds):
        if image_path not in image_results:
            image_results[image_path] = []
        image_results[image_path].append(pred)

    for image_path, digits in image_results.items():
        number = int(''.join(str(d) for d in digits))
        results[image_path] = number

    return results

import os

def recognize_images_in_folder(folder_path):
    """
    识别文件夹中所有图片的数字内容
    参数:
        folder_path: 包含图片的文件夹路径
        model: 加载好的模型
    返回:
        results: 一个字典，键是图片文件名，值是识别结果
    """
    results = {}
    df=pd.read_csv(f'{folder_path}/labels.csv')

    predict_labels=[]
    for _,row in tqdm.tqdm(df.iterrows(),total=len(df)):
        image_path = f"{row['filename']}"
        pred = recognize_image(image_path)
        predict_labels.append(pred)
    df['predict'] = predict_labels
    df['correct'] = df['predict'] == df['label']
    accuracy = df['correct'].mean()
    wrong_samples = df[~df['correct']]

    # 打印指标
    print(f"识别准确率：{accuracy * 100:.2f}%")
    print(f"错误样本数：{len(wrong_samples)}")

    print(classification_report(df['label'], df['predict']))
    cm = confusion_matrix(df['label'], df['predict'])
    print("混淆矩阵：")
    print(cm)

    # 保存更新后的 CSV 文件
    df.to_csv('dataset_with_predictions.csv', index=False)



# 7. 主程序
if __name__ == "__main__":


    # 图片路径
    image_path = "single_pic/test2.png"
    number = recognize_image(image_path)
    print("识别结果：", number)





