# -*- coding: utf-8 -*-
# @Time    : 2023 06
# @Author  : yicao
import copy

import torch
from torch import optim, nn

from utils import model_utils
from utils.record_util import RecordTest, RecordTrain


class AVGServer:
    def __init__(self, device, test_loader, model, log_file, csv_name, lr, sgd_type='sgd', criterion=None):
        self.device = device
        self.model = model
        self.csv_name = csv_name
        self.lr = lr
        self.optimizer = self.get_optimizer(sgd_type)
        self.optimizer.zero_grad()
        self.test_loader = test_loader
        self.log_file = log_file
        # todo:修改test_number
        self.csv_record = RecordTest(log_file=self.log_file, test_file=csv_name)
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        self.now_round = 0
        self.mod_len = model_utils.get_params_len_from_mod(model)

    def aggregate(self, grad_list):
        # 无实际意义，使optimizer中的梯度不为None
        self.optimizer.zero_grad()
        for data in self.test_loader:
            inputs, labels = data
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            break

        grad_mean = grad_list.mean(0)
        model_utils.put_grads_to_optim_gpu(self.optimizer, grad_mean)
        self.optimizer.step()
        if self.now_round % 3 == 2:
            self.test(self.now_round / 3)
        self.now_round += 1

    def download_model(self):
        return self.model

    def test(self, idx):
        with torch.no_grad():
            test_model = copy.deepcopy(self.model).to(self.device)
            test_model.eval()
            for data in self.test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = test_model(inputs)
                self.csv_record.record_test(outputs, labels)
        self.csv_record.print_test(idx)

    def get_optimizer(self, sgd_type):
        if sgd_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.lr)
        elif sgd_type == 'sgdm':
            return optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        elif sgd_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.lr)
        elif sgd_type == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=self.lr)
        else:
            AssertionError("优化器类型错误")


class Client:
    def __init__(self, train_loader, log_file, csv_record, lr, device, local_sgd=10, criterion=None):
        self.train_loader = train_loader
        self.local_sgd = local_sgd
        self.device = device
        self.log_file = log_file
        self.csv_record = csv_record
        self.lr = lr
        self.mod_len = None
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

    def local_train(self, model):
        client_model = copy.deepcopy(model)
        client_model.to(self.device)
        if self.mod_len is None:
            self.mod_len = model_utils.get_params_len_from_mod(client_model)
        optimizer = optim.SGD(client_model.parameters(), lr=self.lr)
        for _ in range(self.local_sgd):
            inputs, labels = self.train_loader.get_next_batch()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = client_model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            self.csv_record.record_train(loss.item(), outputs, labels)
        return self.get_grad(optimizer)

    def get_grad(self, optimizer):
        grad = model_utils.get_grads_from_optim_gpu(optimizer, self.mod_len)
        return grad
