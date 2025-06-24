import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class ToolNet(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),       # (B, 64, H/2, W/2)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),           # (B, 64, H/4, W/4)

            ResidualBlock(64, 128, downsample=True),
            ResidualBlock(128, 128),
            nn.Dropout(0.3),

            ResidualBlock(128, 256, downsample=True),
            ResidualBlock(256, 256),
            nn.Dropout(0.3),

            ResidualBlock(256, 512, downsample=True),
            ResidualBlock(512, 512),
            nn.AdaptiveAvgPool2d((1, 1))     # (B, 512, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class Bottleneck(nn.Module):
    expansion = 4  # 每个 block 输出通道数是中间通道的 4 倍

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample  # 如果维度不匹配，用来调整残差路径

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))  # 1x1
        out = self.relu(self.bn2(self.conv2(out)))  # 3x3
        out = self.bn3(self.conv3(out))  # 1x1

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet50Mine(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50Mine, self).__init__()

        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # [B,64,112,112]
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # [B,64,56,56]

        # 四个阶段，每个 block 的数量分别是 [3, 4, 6, 3]
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        # 平均池化 & 全连接分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # [B,2048,1,1]
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            # 调整 shortcut 的维度
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))  # 第一个 block 可能降采样
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))  # 之后的 block stride=1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # [B,64,112,112]
        x = self.maxpool(x)                     # [B,64,56,56]

        x = self.layer1(x)  # [B,256,56,56]
        x = self.layer2(x)  # [B,512,28,28]
        x = self.layer3(x)  # [B,1024,14,14]
        x = self.layer4(x)  # [B,2048,7,7]

        x = self.avgpool(x)  # [B,2048,1,1]
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 🧪 示例用法
if __name__ == "__main__":
    model = ResNet50Mine(num_classes=15)
    print(model)

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
        filename="plot/tool/structure/model_architecture",
        format="png",
        cleanup=True
    )
