# 构建 VGGNet16 网络模型
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import transforms


class VGGNet16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGGNet16, self).__init__()

        self.Conv1 = nn.Sequential(
            # CIFAR10 数据集是彩色图 - RGB三通道, 所以输入通道为 3, 图片大小为 32*32
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            # inplace-选择是否对上层传下来的tensor进行覆盖运算, 可以有效地节省内存/显存
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 池化层
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 全连接层
        self.fc = nn.Sequential(

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            # 使一半的神经元不起作用，防止参数量过大导致过拟合
            # nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 五个卷积层
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Conv5(x)

        # 数据平坦化处理，为接下来的全连接层做准备
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


class MyAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MyAlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(3, 3), stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=1, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 1024), nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(1024, 1024), nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


if __name__ == '__main__':
    summary(VGGNet16(), (3, 32, 32))  # Params size (MB): 56.80
    summary(MyAlexNet(), (3, 32, 32))  # Params size (MB): 27.22
