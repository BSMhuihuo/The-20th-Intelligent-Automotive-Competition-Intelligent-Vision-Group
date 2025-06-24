import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms,models
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import shutil

# ==== 1. 自定义复杂 CNN 网络结构（ResBlock + 多卷积 + Dropout）====
from tool_model import ToolNet,ResNet50Mine

# ==== 2. 训练相关设置 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 100 #20
batch_size = 32
lr = 1e-4
num_classes = 15
img_size = 224

# ==== 3. 数据预处理与加载 ====
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.ImageFolder("data/TOOL", transform=transform)
class_names=dataset.classes
print(class_names)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ==== 4. 模型初始化、优化器、损失函数 ====
official_resnet = models.resnet50(pretrained=True)

# model = ToolNet(num_classes=num_classes).to(device)
model =ResNet50Mine(num_classes=num_classes).to(device)
# model = models.resnet50(pretrained=True).to(device)
# model = models.resnet50(pretrained=False).to(device)

# 获取你模型的 state_dict
my_model_dict = model.state_dict()

name=f"{model.__class__.__name__}3"
# 从官方模型获取预训练参数
if os.path.exists(f"model/TOOL/{name}.pth")==False:
    pretrained_dict = official_resnet.state_dict()
    print("正在加载预训练参数！")
else:
    pretrained_dict = torch.load(f"model/TOOL/{name}.pth")
    print("正在加载之前训练的参数！")

# 过滤掉不匹配的层（如 fc 层）
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in my_model_dict and v.size() == my_model_dict[k].size()}

# 更新你模型的参数
my_model_dict.update(pretrained_dict)

# 加载更新后的参数

model.load_state_dict(my_model_dict)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ==== 5. 训练和验证 ====
train_loss_list, val_acc_list = [], []
best_acc=0

log_dir = f'runs/{name}'
# 如果路径存在（不管是文件还是文件夹），直接删掉
if not os.path.exists(log_dir):
    # 重新创建空目录
    os.makedirs(log_dir, exist_ok=True)
    print(f"创建新目录：{log_dir}")

tensorboard_writer = SummaryWriter(log_dir=log_dir)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_loss_list.append(total_loss / len(train_loader))

    # 验证精度
    model.eval()
    correct, total = 0, 0

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            # 统计准确率
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 存储预测和真实标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = correct / total
    val_acc_list.append(val_acc)
    if val_acc>best_acc:
        best_acc=val_acc
        torch.save(model.state_dict(), f"model/TOOL/{name}_best.pth")
        print(f"保存了最好的准确率模型,acc:{best_acc}")

    # ⭐ 计算额外指标
    precision = precision_score(all_labels, all_preds, average='macro')  # 或 'weighted'
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    conf_matrix = classification_report(all_labels, all_preds,target_names=class_names, digits=4)

    print(f"Val Accuracy: {val_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print("Confusion Matrix:\n", conf_matrix)
    print(f"Epoch {epoch+1}: Loss={total_loss:.4f}")
    tensorboard_writer.add_scalar('Loss/train', total_loss, epoch)
    tensorboard_writer.add_scalar('val/Accuracy', val_acc, epoch)
    tensorboard_writer.add_scalar('val/Precision', precision, epoch)
    tensorboard_writer.add_scalar('val/Recall', recall, epoch)
    tensorboard_writer.add_scalar('val/F1', f1, epoch)
    tensorboard_writer.add_text('Confusion Matrix', conf_matrix, epoch)


    # ==== 6. 模型保存 ====
    torch.save(model.state_dict(), f"model/TOOL/{name}.pth")
tensorboard_writer.close()
# ==== 7. 可视化 ====
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_acc_list, label='Val Acc')
plt.legend()
plt.title("Training Loss & Validation Accuracy")
plt.savefig("training_result2.png")
plt.show()
