# -*- coding: utf-8 -*-
# @Time    : 2022 09
# @Author  : yicao
# 分布式训练各个客户端的train_loader
import math

from torch.utils.data import DataLoader


class TrainLoader:
    def __init__(self, train_set, batch_size, shuffle=False):
        self.train_set = train_set
        self.batch_size = batch_size
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=shuffle)
        self.iter = iter(self.train_loader)
        self.current_batch = 0
        self.total_batch = math.ceil(len(train_set) / batch_size)

    def get_next_batch(self):
        if self.current_batch == self.total_batch:
            self.current_batch = 0
            self.iter = iter(self.train_loader)
        self.current_batch += 1
        return next(self.iter)

