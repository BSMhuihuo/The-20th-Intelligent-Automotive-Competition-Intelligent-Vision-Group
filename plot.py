from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

# 提取特征向量的函数
from torchvision.datasets import ImageFolder

from tool_model import ToolNet,ResNet50Mine
import torch
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F


def visualize_tsne(features, labels, class_names, save_path=None):
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=reduced[:, 0],
        y=reduced[:, 1],
        hue=[class_names[i] for i in labels],
        palette='tab10',
        s=30,
        ax=ax
    )

    ax.set_title("t-SNE Visualization of Features")

    # 扩展横坐标空间放置图例
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    ax.set_xlim(x_min, x_max + 0.1 * x_range)

    # --- 图例排序 ---
    handles, labels_ = ax.get_legend_handles_labels()

    # 将 handles 和 labels_ 按 labels_ 中标签字符串升序排序
    sorted_pairs = sorted(zip(labels_, handles), key=lambda x: int(x[0]))
    sorted_labels, sorted_handles = zip(*sorted_pairs)

    # 重新设置 legend
    ax.legend(sorted_handles, sorted_labels, loc='upper right', title='Class')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)




def tool_grad_cam(image_name):
    num_classes = 15
    img_size = 224
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50Mine(num_classes=num_classes).to(device)
    pretrained_dict = torch.load(f"model/TOOL/{model.__class__.__name__}best.pth")
    model.load_state_dict(pretrained_dict)


    image = Image.open(image_name).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    input_tensor = input_tensor.to(device)
    model.eval()

    target_layers = [model.layer1, model.layer2, model.layer3, model.layer4]

    for index,target_layer in enumerate(target_layers):
        # target_layer = model.layer4[-1]  # 根据模型结构选择
        cam = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=input_tensor)[0]  # 获取第一张图的热力图

        # 原图用于叠加热力图
        rgb_img = np.array(image.resize((224, 224))).astype(np.float32) / 255
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        plt.imshow(cam_image)
        plt.axis('off')  # 可选：关闭坐标轴
        plt.show()

        cv2.imwrite(f"plot/tool/cam/tool_grad_cam{index}.jpg", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from num_model import ComplexCNN  # 你自己的模型类

def generate_mnist_cam(target_class=None, index=0):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MNIST 预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化
    ])

    # 加载测试集
    test_dataset = datasets.MNIST(root='data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 获取第 index 张图像和标签
    input_tensor, label = test_dataset[index]
    input_tensor = input_tensor.unsqueeze(0).to(device)  # shape: (1, 1, 28, 28)

    # 原始图像（反标准化，用于可视化）
    image_for_cam = input_tensor.squeeze().cpu().numpy() * 0.3081 + 0.1307
    image_for_cam = np.stack([image_for_cam] * 3, axis=-1)  # shape: (28, 28, 3)
    image_for_cam = np.clip(image_for_cam, 0, 1).astype(np.float32)

    # 加载模型
    model = ComplexCNN().to(device)
    model.load_state_dict(torch.load(f'model/Minist/{model.__class__.__name__}model.pth', map_location='cpu'))
    model.eval()

    # 设定目标层（最后一个卷积层）
    target_layer = model.conv2[3]  # Conv2d 层，确保这个索引是你需要的卷积层

    # GradCAM 对象
    cam = GradCAM(model=model, target_layers=[target_layer])

    # 如果未指定 target_class，则使用模型预测结果
    if target_class is None:
        with torch.no_grad():
            output = model(input_tensor)
            target_class = int(torch.argmax(output))

    targets = [ClassifierOutputTarget(target_class)]

    # 生成 CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # shape: (28, 28)

    # 叠加 CAM 到原图
    cam_image = show_cam_on_image(image_for_cam, grayscale_cam, use_rgb=True)

    # 显示
    plt.imshow(cam_image)
    plt.title(f"Grad-CAM for class {target_class} (True label: {label})")
    plt.axis('off')
    plt.show()




def extract_features_tool(model, dataloader):
    features = []
    labels = []

    model.eval()

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="提取特征中"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward 全模型，然后从 logits 前一层提特征
            x = inputs
            for name, module in model.named_children():
                if name == 'classifier' or name == 'fc':  # 忽略最后分类层
                    break
                x = module(x)

            x = torch.flatten(x, 1)  # shape: (B, D)
            features.append(x.cpu())
            labels.append(targets.cpu())

    return torch.cat(features).numpy(), torch.cat(labels).numpy()
def tool_tsne_plot():
    num_classes = 15
    img_size = 224
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    model = ResNet50Mine(num_classes=num_classes).to(device)
    pretrained_dict = torch.load(f"model/TOOL/{model.__class__.__name__}best.pth")
    model.load_state_dict(pretrained_dict)
    class_names = ['1_wrench', '2_soldering_iron', '3_electrodrill', '4_tape_measure', '5_screwdriver', '6_pliers',
                   '7_oscillograph', '8_multimeter', '9_printer','10_keyboard', '11_mobile_phone', '12_mouse', '13_headphones', '14_monitor', '15_speaker']

    val_dataset = ImageFolder("data/TOOL", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    features, labels = extract_features_tool(model, val_loader)
    visualize_tsne(features, labels, class_names,save_path="plot/tool/tsne/tsne.jpg")
def extract_features_number(model, dataloader, device):

    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = images.to(device)
            targets = targets.to(device)

            # 正向传播直到 flatten（不经过 fc1 和 fc2）
            x = model.conv1(images)
            x = model.conv2(x)
            x = x.view(x.size(0), -1)  # 展平成 (batch, 64*7*7)

            features.append(x.cpu())
            labels.append(targets.cpu())

    features = torch.cat(features, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return features, labels
def extract_features_struct(model, val_loader, device):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for data, target in tqdm(val_loader):
            data = data.to(device)

            # 手动前向传播到 fc1 激活之后（不调用 model(data)）
            x = model.pool(F.relu(model.conv1(data)))
            x = model.pool(F.relu(model.conv2(x)))
            x = model.pool(F.relu(model.conv3(x)))
            x = x.view(-1, 128 * 28 * 28)
            x = F.relu(model.fc1(x))  # 512维特征（无 dropout）
            # x = model.dropout(F.relu(model.fc1(x)))
            features.append(x.cpu())
            labels.append(target)

    features_tensor = torch.cat(features, dim=0)
    labels_tensor = torch.cat(labels, dim=0)
    return features_tensor, labels_tensor
def number_tsne_plot():

    from num_model import ComplexCNN  # 你自己的模型类


    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MNIST 预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化
    ])

    # 加载测试集
    test_dataset = datasets.MNIST(root='data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 加载模型
    model = ComplexCNN().to(device)
    model.load_state_dict(torch.load(f'model/Minist/{model.__class__.__name__}model.pth', map_location=torch.device('cpu'))) # 加载模型
    model.eval()

    # 可视化函数已定义
    # 直接运行：
    features, labels = extract_features_number(model, test_loader,device)
    class_names = [str(i) for i in range(10)]
    visualize_tsne(features, labels, class_names,save_path="plot/number/tsne/tsne.jpg")
def visualize_tsne_struct(features, labels, class_names, save_path=None):
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=reduced[:, 0],
        y=reduced[:, 1],
        hue=[class_names[i] for i in labels],
        palette='tab10',
        s=30,
        ax=ax
    )

    ax.set_title("t-SNE Visualization of Features")

    # 扩展横坐标空间放置图例
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    ax.set_xlim(x_min, x_max + 0.1 * x_range)

    # --- 图例排序 ---
    handles, labels_ = ax.get_legend_handles_labels()

    # 将 handles 和 labels_ 按 labels_ 中标签字符串升序排序
    sorted_pairs = sorted(zip(labels_, handles), key=lambda x: x[0])
    sorted_labels, sorted_handles = zip(*sorted_pairs)

    # 重新设置 legend
    ax.legend(sorted_handles, sorted_labels, loc='upper right', title='Class')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
def struct_tsne_plot():
    from struct_model import MyCNN
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 超参数
    BATCH_SIZE = 16
    EPOCHS = 20
    LR = 1e-4

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    val_dataset = datasets.ImageFolder("data/STRUCT/val", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # 模型与优化器
    model = MyCNN().to(device)
    model.load_state_dict(
        torch.load(f'model/STRUCT/tool_vs_digit_best3.pth', map_location=torch.device('cpu')))  # 加载模型
    # 加载模型
    model.eval()

    # 可视化函数已定义
    # 直接运行：
    features, labels = extract_features_struct(model, val_loader,device)
    class_names = ["DIGIT","TOOL"]
    visualize_tsne_struct(features, labels, class_names,save_path="plot/struct/tsne/tsne.jpg")


if __name__ == "__main__":
    # tool_tsne_plot()
    tool_grad_cam("data/STRUCT/val/tool/mouse_090.jpg")
    # number_tsne_plot()
    # generate_mnist_cam(index=3)
    # struct_tsne_plot()