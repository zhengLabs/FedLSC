# -*- coding: utf-8 -*-
# @Time    : 2023 02
# @Author  : yicao

import copy
import math
import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from models import IMDBDataSet, LSTM
from utils import model_utils, public_utils, log_util, SPFSManager, fed_avg_top_k
from utils.record_util import RecordAccUtil
from utils.train_loader_distribute import TrainLoader

MAX_LEN = 300

# 计算预测准确性
def accuracy(y_pred, y_true):
    label_pred = y_pred.max(dim=1)[1]
    acc = len(y_pred) - torch.sum(torch.abs(label_pred-y_true)) # 正确的个数
    return acc.detach().cpu().numpy() / len(y_pred)

def train(seed = 50, client_num = 15, batch_size = 100, epoch = 200, local_sgd = 10, lr = 0.5
          , dataset ='IMDB', distribution = 'iid', class_num = 2, mod_name ='LSTM', check = 5, cuda=0, sparse=0.1):
    train_start_time = time.time()
    first_agg_time = None
    public_utils.set_seed(seed)

    # 记录日志和实验数据
    log_name = f"{os.path.basename(__file__)[:-3]} {dataset}-{distribution} {mod_name}" \
                f" norm sparse={sparse} lr={lr} n={client_num} check={check} seed={seed}"
    csv_dir = os.path.join('./results', f'{dataset}-{distribution}', f'{mod_name}')
    csv_name = os.path.join(csv_dir, f'{os.path.basename(__file__)[:-3]} sparse={sparse} lr={lr} n={client_num} check={check} seed={seed}')

    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    log_file = log_util.create_log(f"{log_name}")

    device = (torch.device(f'cuda:{cuda}') if torch.cuda.is_available() else torch.device('cpu'))

    train_path = './data/train.txt'  # 预处理后的训练集文件地址
    test_path = './data/test.txt'  # 预处理后的训练集文件地址
    vocab = np.load('./data/vocab.npy', allow_pickle=True).item()  # 加载本地已经存储的vocab

    train_set = IMDBDataSet.IMDBDataSet(text_path=train_path, vocab=vocab, MAX_LEN=MAX_LEN)
    test_set = IMDBDataSet.IMDBDataSet(text_path=test_path, vocab=vocab, MAX_LEN=MAX_LEN)

    log_util.log(log_file, f"train-size: {len(train_set)},  test-size: {len(test_set)}")

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
    residual = torch.zeros([client_num, mod_len])
    bitmaps = torch.empty([client_num, mod_len])

    sub_set_len = len(train_set) // client_num
    communication_num = math.ceil(epoch * (sub_set_len / batch_size / local_sgd))
    test_time = 3

    record_utils = RecordAccUtil(batch_size=batch_size, train_number=len(train_set), test_number=len(test_set),
                                 print_batch=local_sgd * client_num,log_file=log_file, csv_name=csv_name)

    # 初始化APFManager
    spfs_manager = SPFSManager.SPFSManager(mod_len)
    topk = fed_avg_top_k.FedAvgTopK(mod_len, sparse)

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
            new_param = spfs_manager.get_params(optimizer)
            residual[idx] *= spfs_manager.freezing_bitmap
            bitmaps[idx], params[idx], residual[idx] = topk.get_params_residual(new_param, residual[idx])

        # 聚合参数，更新模型
        if topk.last_params is None:
            param_mean = params.mean(0)
            first_agg_time = time.time()
            log_util.log(log_file, f"first aggregate finished!")
        else:
            param_mean = model_utils.sparse_aggregate_params(params, topk.last_params, bitmaps)
        topk.last_params = param_mean
        model_utils.params2mod(param_mean, center_model)
        log_util.log(log_file, f"the {i} communication round aggregate finished! current time {round(time.time() - first_agg_time, 3)}s")


        if i % check == check - 1:
            spfs_manager.check(param_mean)
            log_util.log(log_file, f"冻结稀疏度：{round(100. * torch.count_nonzero(spfs_manager.freezing_bitmap).item() / mod_len, 2)}%")

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








