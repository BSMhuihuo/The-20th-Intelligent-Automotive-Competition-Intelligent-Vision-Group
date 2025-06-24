import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10个输出对应0-9数字

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出: 32 x 14 x 14
            nn.Dropout(0.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出: 64 x 7 x 7
            nn.Dropout(0.25)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Linear(256, 10)  # 10类输出

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.fc2(x)
        return x
if  __name__ == '__main__':
    model = ComplexCNN()
    # 创建一个 dummy 输入（形状与真实输入一致）
    dummy_input = torch.randn(1, 1, 28, 28)


    # writer = SummaryWriter(f"runs/{model.__class__.__name__}_struct")
    # # 添加模型结构图
    # writer.add_graph(model, dummy_input)
    #
    # writer.close()

    # from torchviz import make_dot
    # # 假设你的模型是 model，输入是 dummy_input
    # output = model(dummy_input)
    # make_dot(output, params=dict(model.named_parameters())).render("model_architecture.gv", format="png")

    from torchview import draw_graph

    model.eval()

    # 使用 torchview 创建图
    graph = draw_graph(
        model,
        input_data=dummy_input,
        expand_nested=True,
        graph_name="model_architecture",
        roll=True  # 横向布局
    )

    # 设置图像属性：DPI、方向等
    graph.visual_graph.attr(dpi='300')  # 设置高清
    graph.visual_graph.attr(rankdir='LR')  # 设置横向（Left to Right）

    # 渲染并保存为 PNG
    graph.visual_graph.render(
        filename="plot/number/structure/model_architecture",
        format="png",
        cleanup=True
    )

