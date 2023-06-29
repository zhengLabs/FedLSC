# -*- coding: utf-8 -*-
# @Time    : 2022 09
# @Author  : yicao
import csv
import os
import time

import numpy as np
import torch
import math

from utils import log_util


class RecordTest:
    def __init__(self, log_file, test_file):
        self.log_file = log_file
        self.test_file = test_file + ' test.csv'
        self.test_acc_sum = 0.0
        self.test_cnt = 0
        with open(self.test_file, 'w') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(['epoch', 'Test Acc'])
        print(f"创建记录文件：{self.test_file}")

    def record_test(self, outputs, labels):
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc = 100. * correct / labels.size(0)
        self.test_acc_sum += acc
        self.test_cnt += 1

    def print_test(self, epoch=0):
        test_acc = round((self.test_acc_sum / self.test_cnt), 4)
        self.log("---------------------Test---------------------")
        self.log(f"Test Acc: {test_acc}%")
        self.log("---------------------Test---------------------")
        with open(self.test_file, 'a+') as f:
            csv_write = csv.writer(f)
            csv_write.writerow([epoch, test_acc])
        self.test_acc_sum, self.test_cnt = 0, 0
        return test_acc

    def log(self, line):
        log_util.log(self.log_file, line)


class RecordTrain:
    def __init__(self, log_file, train_file, print_freq):
        self.acc_sum = 0
        self.loss_sum = 0
        self.cnt = 0
        self.log_file = log_file
        self.train_file = train_file + ' train.csv'
        self.print_freq = print_freq
        self.last_time = time.time()

    def record_init(self):
        self.acc_sum = 0
        self.loss_sum = 0
        self.last_time = time.time()

    def record_train(self, loss, outputs, labels, msg="", epoch=0):
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc = 100. * correct / labels.size(0)

        self.acc_sum += acc
        self.loss_sum += loss

        if self.cnt % self.print_freq == (self.print_freq - 1):
            self.log(
                f"{msg}epoch: {epoch}, iter: {self.cnt}, acc: {round(self.acc_sum / self.print_freq, 4)}%,"
                f" loss: {round(self.loss_sum / self.print_freq, 4)}, time: {round(time.time() - self.last_time, 3)}s")
            with open(self.train_file, 'a+') as f:
                csv_write = csv.writer(f)
                csv_write.writerow([epoch, self.cnt, round(self.acc_sum / self.print_freq, 4),
                                    round(self.loss_sum / self.print_freq, 4)])
            self.record_init()
        self.cnt += 1

    def log(self, line):
        log_util.log(self.log_file, line)


class RecordAccUtil:
    def __init__(self, batch_size, train_number=1, test_number=1, print_batch=100, log_file="", csv_name=None):
        self.log_file = log_file
        self.csv_name = csv_name
        self.loss_list = []  # 存储每个batch的loss
        self.acc_list = []  # 存储每个batch的acc
        self.epoch = 0
        self.total_iter = 0  # 当前迭代次数
        self.batch_size = batch_size
        self.train_number = train_number  # 训练集大小
        self.test_number = test_number  # 测试集大小
        self.epoch_batch_size = math.ceil(self.train_number / self.batch_size)  # 一个epoch的batch_size数
        self.print_batch = print_batch  # 打印周期
        self.print_train_acc = 0
        self.print_loss = 0
        self.print_test_acc = 0
        self.last_time = time.time()

        self.train_csv_name = csv_name + ' train.csv'
        self.test_csv_name = csv_name + ' test.csv'
        with open(self.train_csv_name, 'w') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(['epoch', 'iter', 'acc', 'loss'])
        with open(self.test_csv_name, 'w') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(['epoch', 'Test Acc'])
        print(f"创建记录文件：{self.train_csv_name}")
        print(f"创建记录文件：{self.test_csv_name}")

    def record_train(self, loss, outputs, labels, msg="", epoch=0):
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc = 100. * correct / labels.size(0)
        self.acc_list.append(acc)
        self.loss_list.append(loss)

        self.print_train_acc += acc
        self.print_loss += loss

        if self.total_iter % self.print_batch == (self.print_batch - 1):
            self.log(
                f"{msg}epoch: {self.epoch}, iter: {self.total_iter}, acc: {round(self.print_train_acc / self.print_batch, 4)}%,"
                f" loss: {round(self.print_loss / self.print_batch, 4)}, time: {round(time.time() - self.last_time, 3)}s")
            with open(self.train_csv_name, 'a+') as f:
                csv_write = csv.writer(f)
                csv_write.writerow([epoch, self.total_iter, round(self.print_train_acc / self.print_batch, 4),
                                    round(self.print_loss / self.print_batch, 4)])
            self.print_train_acc, self.print_loss = 0, 0
            self.last_time = time.time()

        self.total_iter += 1
        self.epoch = int(self.total_iter / self.epoch_batch_size)

    def record_train_nll(self, loss, outputs, labels, msg="", epoch=0):
        label_pred = outputs.max(dim=1)[1]
        correct = len(outputs) - torch.sum(torch.abs(label_pred - labels))  # 正确的个数
        acc = (100. * correct / labels.size(0)).item()
        self.acc_list.append(acc)
        self.loss_list.append(loss)

        self.print_train_acc += acc
        self.print_loss += loss

        if self.total_iter % self.print_batch == (self.print_batch - 1):
            self.log(
                f"{msg}epoch: {self.epoch}, iter: {self.total_iter}, acc: {round(self.print_train_acc / self.print_batch, 4)}%,"
                f" loss: {round(self.print_loss / self.print_batch, 4)}, time: {round(time.time() - self.last_time, 3)}s")
            with open(self.train_csv_name, 'a+') as f:
                csv_write = csv.writer(f)
                csv_write.writerow([epoch, self.total_iter, round(self.print_train_acc / self.print_batch, 4),
                                    round(self.print_loss / self.print_batch, 4)])
            self.print_train_acc, self.print_loss = 0, 0
            self.last_time = time.time()

        self.total_iter += 1
        self.epoch = int(self.total_iter / self.epoch_batch_size)

    def record_test(self, outputs, labels):
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc = 100. * correct / labels.size(0)
        self.print_test_acc += acc

    def record_test_nnl(self, outputs, labels):
        label_pred = outputs.max(dim=1)[1]
        acc = len(outputs) - torch.sum(torch.abs(label_pred - labels))  # 正确的个数
        self.print_test_acc += acc.detach().cpu().numpy()

    def print_test_accuracy(self, epoch):
        test_acc = round((self.print_test_acc / self.test_number) * 100, 4)
        self.log("---------------------Test---------------------")
        self.log(f"Test Acc: {test_acc}%")
        self.log("---------------------Test---------------------")
        with open(self.test_csv_name, 'a+') as f:
            csv_write = csv.writer(f)
            csv_write.writerow([epoch, test_acc])
        self.print_test_acc = 0
        return test_acc

    def log(self, line):
        log_util.log(self.log_file, line)


class RecordCompressUtil:
    def __init__(self, model_length):
        self.total_compress = 0  # 总压缩率
        self.total_idx = 0
        self.epoch_compress = 0  # 单个epoch的压缩率
        self.epoch_idx = 0
        self.model_length = model_length

    def record_compress(self, bitmaps: np.ndarray):
        self.epoch_compress += np.count_nonzero(bitmaps)
        self.epoch_idx += 1

    def epoch_end(self):
        log_util.log(self.epoch_compress)
        log_util.log(self.epoch_idx)
        compress = self.epoch_compress / self.epoch_idx
        log_util.log(compress)
        log_util.log(f"epoch训练结束， 平均稀疏率为：{round(compress * 100 / self.model_length, 2)}%")
        self.epoch_compress = 0
        self.epoch_idx = 0
        self.total_compress += compress
        self.total_idx += 1

    def train_end(self):
        compress = self.total_compress / self.total_idx
        log_util.log(f"全部训练结束， 平均稀疏率为：{round(compress * 100 / self.model_length, 2)}%")


class RecordCompressLayerUtil:
    def __init__(self, optim: torch.optim.Optimizer, mod_len, print_size, create_csv=False, csv_name=None):
        self.mod_len = mod_len
        self.layer_size = []
        self.m = 0
        self.print_size = print_size
        self.print_t = 0
        self.create_csv = create_csv
        self.csv_name = csv_name

        # 计算模型层数m和各层的参数量q
        for param in optim.param_groups[0]['params']:
            params = param.data.view(-1)
            self.m += 1
            self.layer_size.append(len(params))
        self.rs = np.zeros(self.m)
        self.layer_size = np.array(self.layer_size)

        if self.create_csv & (self.csv_name is not None):
            with open(os.path.join('results', self.csv_name), 'w') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(range(self.m))

    def record_compress(self, bitmap: np.ndarray):
        # 根据bitmap记录本轮各层传递的参数量
        start = 0
        for idx, x in enumerate(self.layer_size):
            self.rs[idx] += np.count_nonzero(bitmap[start:start + x])
            start += x
        self.print_t += 1

        if self.print_size == self.print_t:
            self.print_t = 0
            self.rs = self.rs / self.print_size / self.layer_size
            print_list = list(map(lambda k: round(k, 3) if round(k, 3) != 0 else k, self.rs))
            self.rs = np.zeros(self.m)
            # log_util.log(f"各层稀疏率为: {print_list}")
            if self.create_csv & (self.csv_name is not None):
                with open(os.path.join('results', self.csv_name), 'a+') as f:
                    csv_write = csv.writer(f)
                    csv_write.writerow(print_list)
