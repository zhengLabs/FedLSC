# -*- coding: utf-8 -*-
# @Time    : 2023 02
# @Author  : yicao

import copy
import csv
import os

import math
import time

import numpy as np
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from models import MyResNet, CNN, LeNet5, IMDBDataSet, LSTM
from utils import model_utils, top_k_utils, public_utils, APFManager, log_util
from utils.record_util import RecordAccUtil, RecordCompressUtil, RecordCompressLayerUtil
from utils.train_loader_distribute import TrainLoader

MAX_LEN = 300


# 计算预测准确性
def accuracy(y_pred, y_true):
    label_pred = y_pred.max(dim=1)[1]
    acc = len(y_pred) - torch.sum(torch.abs(label_pred-y_true)) # 正确的个数
    return acc.detach().cpu().numpy() / len(y_pred)

def train(seed = 50, client_num = 15, batch_size = 100, epoch = 200, local_sgd = 10, lr = 0.5
          , dataset ='IMDB', distribution = 'iid', class_num = 2, mod_name ='LSTM', check = 0, cuda=0):
    local_sgd = 1
    train_start_time = time.time()
    public_utils.set_seed(seed)

    # 记录日志和实验数据
    log_name = f"{os.path.basename(__file__)[:-3]} {dataset}-{distribution} {mod_name}" \
                f" norm lr={lr} n={client_num} check={check}"
    csv_dir = os.path.join('./results', f'{dataset}-{distribution}', f'{mod_name}')
    csv_name = os.path.join(csv_dir, f'{os.path.basename(__file__)[:-3]} lr={lr} n={client_num} check={check}')

    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    log_file = log_util.create_log(f"{log_name}")

    device = (torch.device(f'cuda:{cuda}') if torch.cuda.is_available() else torch.device('cpu'))

    train_path = './data/train.txt'  # 预处理后的训练集文件地址
    test_path = './data/test.txt'  # 预处理后的训练集文件地址
    vocab = np.load('./data/vocab.npy', allow_pickle=True).item()  # 加载本地已经存储的vocab

    train_set = IMDBDataSet.IMDBDataSet(text_path=train_path, vocab=vocab, MAX_LEN=MAX_LEN)
    test_set = IMDBDataSet.IMDBDataSet(text_path=test_path, vocab=vocab, MAX_LEN=MAX_LEN)

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=5)
    train_loader_list = []
    sub_set_length = len(train_set) // client_num
    for idx in range(client_num):
        sub_set = Subset(train_set, range(idx * sub_set_length, (idx + 1) * sub_set_length))
        train_loader_list.append(TrainLoader(sub_set, batch_size=batch_size, shuffle=True))

    # 初始化全局模型
    center_model = LSTM.LSTM(input_size=len(vocab), embed_size=300, hidden_size=128, num_layers=2)  # 定义模型
    criterion = nn.NLLLoss()
    mod_len = model_utils.get_params_len_from_mod(center_model)
    log_util.log(log_file, f"模型总参数：{mod_len}")

    #  记录客户端传过来的参数
    params = torch.empty([client_num, mod_len])

    sub_set_len = len(train_set) // client_num
    communication_num = math.ceil(epoch * (sub_set_len / batch_size / local_sgd))
    test_time = 30

    record_utils = RecordAccUtil(batch_size=batch_size, train_number=len(train_set), test_number=len(test_set),
                                 print_batch=local_sgd * client_num,log_file=log_file, csv_name=csv_name)

    for i in range(communication_num):
        # 客户端接收集中模型，因为接收的模型都是一样，为了节省性能，模拟多个客户端共用一个模型
        for idx in range(client_num):
            client_model = copy.deepcopy(center_model)
            client_model = client_model.to(device)
            optimizer = optim.SGD(client_model.parameters(), lr=lr)
            # 在本地训练10次后同步
            for j in range(local_sgd):
                # 使用一个batch的数据进行训练
                inputs, labels = train_loader_list[idx].get_next_batch()
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = client_model(inputs)

                loss = criterion(outputs.log(), labels)
                loss.backward()
                optimizer.step()
                # print(accuracy(outputs, labels))
                record_utils.record_train_nll(loss.item(), outputs, labels)
            params[idx] = model_utils.get_params_tensor(optimizer, mod_len)

        # 聚合参数，更新模型
        grad_mean = params.mean(0)
        model_utils.params2mod(grad_mean, center_model)

        # 每100个通信轮进行一次测试
        if i % test_time == test_time - 1:
            with torch.no_grad():
                test_model = copy.deepcopy(center_model)
                test_model = test_model.to(device)
                for data in test_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = test_model(inputs)
                    record_utils.record_test_nnl(outputs, labels)

                record_utils.print_test_accuracy(i // test_time)

    log_util.log(log_file, f"训练结束，耗时：{round(time.time() - train_start_time, 3)}s")







