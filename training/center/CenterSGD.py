# -*- coding: utf-8 -*-
# @Time    : 2023 05
# @Author  : yicao
import os
import sys
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# sys.path.append('../../')
from utils import model_utils, public_utils, log_util
from utils.record_util import RecordAccUtil


class CenterSGD:
    def __init__(self, dataset, device, model, batch_size, log_file, csv_name, lr):
        train_set, test_set = dataset
        self.lr = lr
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
        self.device = device
        self.model = model.to(device)
        self.optimizer = self.get_optimizer()
        self.criterion = self.get_criterion()
        self.mod_len = model_utils.get_params_len_from_mod(self.model)
        self.log_file = log_file
        self.csv_record = RecordAccUtil(batch_size=batch_size, train_number=len(train_set), test_number=len(test_set),
                                        print_batch=50, log_file=log_file, csv_name=csv_name)

    def get_optimizer(self):
        return optim.SGD(self.model.parameters(), lr=self.lr)

    @staticmethod
    def get_criterion():
        return nn.CrossEntropyLoss()

    def _train(self):
        for idx, data in enumerate(self.train_loader):
            self.model.train()
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.csv_record.record_train(loss.item(), outputs, labels)

    def __test(self, idx):
        with torch.no_grad():
            self.model.eval()
            for data in self.test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                self.csv_record.record_test(outputs, labels)
        self.csv_record.print_test_accuracy(idx)

    def start_training(self, epoch):
        train_start_time = time.time()
        for i in range(epoch):
            epoch_start_time = time.time()
            self._train()
            self.__test(i)
            log_util.log(self.log_file, f"第{i}个epoch结束，耗时：{round(time.time() - epoch_start_time, 3)}s")
        log_util.log(self.log_file, f"训练结束，耗时：{round(time.time() - train_start_time, 3)}s")


if __name__ == '__main__':
    os.chdir('/home/jky/zyq/SPFS')
    data_name = "Cifar10"
    mod_name = "LeNet"
    file_name = "CenterAdam"
    cuda = 0

    data_set = public_utils.get_data_set(data_name)
    device = (torch.device(f'cuda:{cuda}') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"device:{device}")
    model = public_utils.get_net(mod_name, 10).to(device)
    log_file = log_util.create_log(f"{file_name} {data_name} {mod_name}")
    csv_path = os.path.join('./A-Result-Center', data_name, mod_name)
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
