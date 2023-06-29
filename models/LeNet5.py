# -*- coding: utf-8 -*-
# @Time    : 2022 07
# @Author  : yicao

import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary


# from torchsummary import summary


class LeNet52(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=0)  # 32*32*3 -> 28*28*6
        # self.conv1 = nn.Conv2d(1, 6, 5, padding=2)  # 28*28*1 -> 28*28*6 -> 14*14*6
        self.conv2 = nn.Conv2d(6, 16, 5)  # 14*14*6 -> 10*10*16 -> 5*5*16
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # MNIST
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=1)  # Cifar10
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (3,32,32) -> (16,28,28)
        x = self.pool1(x)  # (16,28,28) -> (16,14,14)
        x = F.relu(self.conv2(x))  # (16,14,14) -> (32,10,10)
        x = self.pool2(x)  # (32,10,10) -> (32,5,5)
        x = x.view(-1, 32 * 5 * 5)  # (32,5,5) -> 35*5*5
        x = F.relu((self.fc1(x)))  # 120
        x = F.relu((self.fc2(x)))  # 84
        x = self.fc3(x)  # 10
        return x


# if __name__ == '__main__':
#     summary(LeNet5(), (3, 32, 32)) # Params size (MB): 0.46
