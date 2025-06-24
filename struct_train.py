import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from struct_model import MyCNN
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from torch.utils.tensorboard import SummaryWriter
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

train_dataset = datasets.ImageFolder("data/STRUCT/train", transform=transform)
val_dataset = datasets.ImageFolder("data/STRUCT/val", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 模型与优化器
model = MyCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 训练过程
train_acc_list = []
val_acc_list = []
log_dir=f"runs/STRUCT/{model.__class__.__name__}3"
class_names = ["DIGIT","TOOL"]
best_acc = 0
writer = SummaryWriter(log_dir=log_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
for epoch in range(EPOCHS):
    model.train()
    correct, total = 0, 0
    total_loss = 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
    train_acc = correct / total
    train_acc_list.append(train_acc)

    # 验证
    model.eval()
    correct, total = 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 存储预测和真实标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_acc = correct / total
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    val_acc = correct / total
    val_acc_list.append(val_acc)
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f"model/STRUCT/tool_vs_digit_best3.pth")
        print(f"保存了最好的准确率模型,acc:{best_acc}")

    # ⭐ 计算额外指标
    precision = precision_score(all_labels, all_preds, average='macro')  # 或 'weighted'
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    conf_matrix = classification_report(all_labels, all_preds, target_names=class_names, digits=4)

    print(f"Val Accuracy: {val_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print("Confusion Matrix:\n", conf_matrix)
    print(f"Epoch {epoch + 1}: Loss={total_loss:.4f}")
    print(f"测试准确率: {correct / total:.4f}")
    writer.add_scalar("Loss/train", total_loss, epoch)
    writer.add_scalar("val/Accuracy", val_acc, epoch)
    writer.add_scalar("val/Precision", precision, epoch)
    writer.add_scalar("val/Recall", recall, epoch)
    writer.add_scalar("val/F1", f1, epoch)
    writer.add_text("Confusion Matrix", conf_matrix, epoch)

    # 保存模型
    torch.save(model.state_dict(), f"model/STRUCT/tool_vs_digit3.pth")

# 绘图
plt.plot(train_acc_list, label="Train Acc")
plt.plot(val_acc_list, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.legend()
plt.savefig("accuracy_plot.png")
plt.show()
