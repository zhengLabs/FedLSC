# -*- coding: utf-8 -*-
# @Time    : 2023 05
# @Author  : yicao
from torch import optim

from training.center.CenterSGD import CenterSGD


class CenterSGDM(CenterSGD):
    def __init__(self, dataset, device, model, batch_size, log_file, csv_name, lr):
        super().__init__(dataset, device, model, batch_size, log_file, csv_name, lr)

    def get_optimizer(self):
        return optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)


class CenterAdam(CenterSGD):
    def __init__(self, dataset, device, model, batch_size, log_file, csv_name, lr):
        super().__init__(dataset, device, model, batch_size, log_file, csv_name, lr)

    def get_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)


class CenterRMSprop(CenterSGD):
    def __init__(self, dataset, device, model, batch_size, log_file, csv_name, lr):
        super().__init__(dataset, device, model, batch_size, log_file, csv_name, lr)

    def get_optimizer(self):
        return optim.RMSprop(self.model.parameters(), lr=self.lr)
