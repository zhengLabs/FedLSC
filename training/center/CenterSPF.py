# -*- coding: utf-8 -*-
# @Time    : 2023 05
# @Author  : yicao
import torch
from torch import optim

from training.center.CenterSGD import CenterSGD
from utils import model_utils


class SPFManager:
    def __init__(self, mod_len):
        self.EMA = torch.tensor(0.90)  # 动量因子
        self.Ek = torch.tensor(0.0)
        self.Ek_abs = torch.tensor(0.0)
        self.freezing_bitmap = torch.ones(mod_len)  # 冻结位图
        self.Ts = 0.05  # 稳定性阈值
        self.mod_len = mod_len

    def calc_pk(self, grad):
        self.Ek = self.Ek * (1 - self.freezing_bitmap) + (
                self.EMA * self.Ek + (1 - self.EMA) * grad) * self.freezing_bitmap
        self.Ek_abs = self.Ek_abs * (1 - self.freezing_bitmap) + (
                self.EMA * self.Ek_abs + (1 - self.EMA) * abs(grad)) * self.freezing_bitmap
        pk = abs(self.Ek) / (self.Ek_abs + 1e-8)

        return pk

    def check(self, grad):
        pk = self.calc_pk(grad)
        self.freezing_bitmap = torch.where((self.freezing_bitmap == 1) & (pk < self.Ts), torch.tensor(0),
                                           torch.tensor(1))
        self.record_sparse()

    def record_sparse(self):
        print(f"稀疏度：{round((torch.count_nonzero(self.freezing_bitmap) / self.mod_len * 100).item(), 2)}%")


class SPF(CenterSGD):
    def __init__(self, dataset, device, model, batch_size, log_file, csv_name, check_round, lr):
        super().__init__(dataset, device, model, batch_size, log_file, csv_name, lr)
        self.spf = SPFManager(self.mod_len)
        self.check_round = check_round

    def _train(self):
        for idx, data in enumerate(self.train_loader):
            self.model.train()
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.csv_record.record_train(loss.item(), outputs, labels)
            grads = model_utils.get_grads_from_optim_gpu(self.optimizer, self.mod_len)
            grads *= self.spf.freezing_bitmap
            model_utils.put_grads_to_optim_gpu(self.optimizer, grads.to(self.device))
            self.optimizer.step()

            if idx % self.check_round == 0:
                self.spf.check(grads)

    def get_optimizer(self):
        return optim.SGD(self.model.parameters(), lr=self.lr)
