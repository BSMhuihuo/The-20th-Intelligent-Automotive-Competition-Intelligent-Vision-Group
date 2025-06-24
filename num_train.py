import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from num_model import SimpleCNN,ComplexCNN
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from torch.utils.tensorboard import SummaryWriter
import os
# 定义模型

# 参数
batch_size = 64
epochs = 100
learning_rate = 0.001

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值与标准差
])

# 加载 MNIST 数据集
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = SimpleCNN().to(device)
model = ComplexCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_loss_list, val_acc_list = [], []
best_acc = 0
class_names = [str(i) for i in range(10)]
log_dir=f"runs/Minist/{model.__class__.__name__}2"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)
# 训练过程
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    # 测试准确率
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

            # 存储预测和真实标签
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    val_acc = correct / total
    val_acc_list.append(val_acc)
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f"model/Minist/{model.__class__.__name__}best2.pth")
        print(f"保存了最好的准确率模型,acc:{best_acc}")

    # 计算额外指标
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
writer.close()
# 保存模型
torch.save(model.state_dict(), f"model/Minist/{model.__class__.__name__}model2.pth")
print(f"模型已保存为 model/Minist/{model.__class__.__name__}model2.pth")
