import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # (B, 3, 224, 224) -> (B, 32, 224, 224)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B, 32, 112, 112)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 64, 56, 56)
        x = self.pool(F.relu(self.conv3(x)))  # (B, 128, 28, 28)
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
if __name__ == '__main__':
    model = MyCNN()
    # 测试输入
    dummy_input = torch.randn(1, 3, 224, 224)
    y = model(dummy_input)
    print(y.shape)  # 应输出 [1, 1000]

    from torchview import draw_graph

    model.eval()

    # 使用 torchview 创建图
    graph = draw_graph(
        model,
        input_data=dummy_input,
        expand_nested=True,
        graph_name="model_architecture",
    )

    # 设置图像属性：DPI、方向等
    graph.visual_graph.attr(dpi='300')  # 设置高清
    # graph.visual_graph.attr(rankdir='LR')  # 设置横向（Left to Right）

    # 渲染并保存为 PNG
    graph.visual_graph.render(
        filename="plot/struct/structure/model_architecture",
        format="png",
        cleanup=True
    )
