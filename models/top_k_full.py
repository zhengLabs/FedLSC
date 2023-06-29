# -*- coding: utf-8 -*-
# @Time    : 2022 09
# @Author  : yicao
# 28*28-4096-4096-4096-10
# 28*28*4096=3211264 4096*4096=16777216 4096*4096=16777216 4096*10=40960 3211264+16777216+16777216+40960=36806656
# 28*28*512+512*512+512*128+128*10=730368
import torch.nn as nn


class TopKFull(nn.Module):
    def __init__(self):
        super(TopKFull, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class TopKFullSmall(nn.Module):
    def __init__(self):
        super(TopKFullSmall, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

