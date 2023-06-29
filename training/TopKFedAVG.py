# -*- coding: utf-8 -*-
# @Time    : 2022 12
# @Author  : yicao
import copy
import math
import os
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from utils import model_utils, public_utils, fed_avg_top_k, log_util
from utils.record_util import RecordAccUtil
from utils.train_loader_distribute import TrainLoader


def train(seed=50, client_num=16, batch_size=100, epoch=200, local_sgd=10, lr=0.1
          , dataset='Cifar10', distribution='iid', class_num=10, mod_name='LeNet', check=0, cuda=0, sparse=0.4):
    train_start_time = time.time()
    first_agg_time = None
    public_utils.set_seed(seed)

    log_name = f"{os.path.basename(__file__)[:-3]} {dataset}-{distribution} {mod_name}" \
               f" spares={sparse} lr={lr} n={client_num} check={check} seed={seed}"
    csv_dir = os.path.join('./results', f'{dataset}-{distribution}', f'{mod_name}', f"{os.path.basename(__file__)[:-3]}")
    csv_name = os.path.join(csv_dir,
                            f'spares={sparse} lr={lr} n={client_num} check={check} seed={seed}')

    device = (torch.device(f'cuda:{cuda}') if torch.cuda.is_available() else torch.device('cpu'))

    train_set, test_set = public_utils.get_data_set(dataset)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=0)
    train_loader_list = []

    data_part = public_utils.get_data_part(dataset, distribution, train_set.targets, seed, client_num)

    for i in range(client_num):
        sub_set = Subset(train_set, data_part.client_dict[i])
        train_loader_list.append(TrainLoader(sub_set, batch_size=batch_size, shuffle=True))

    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    log_file = log_util.create_log(f"{log_name}")

    # 初始化全局模型
    center_model = public_utils.get_net(mod_name, class_num)
    criterion = nn.CrossEntropyLoss()
    mod_len = model_utils.get_params_len_from_mod(center_model)
    #  记录客户端传过来的参数
    params = torch.empty([client_num, mod_len])
    residual = torch.zeros([client_num, mod_len])
    bitmaps = torch.empty([client_num, mod_len])

    topk = fed_avg_top_k.FedAvgTopK(mod_len, sparse)

    sub_set_len = len(train_set) // client_num
    communication_num = math.ceil(epoch * (sub_set_len / batch_size / local_sgd))
    test_time = 3

    record_utils = RecordAccUtil(batch_size=batch_size, train_number=len(train_set), test_number=len(test_set),
                                 print_batch=local_sgd * client_num, log_file=log_file, csv_name=csv_name)

    for i in range(communication_num):
        for idx in range(client_num):
            client_model = copy.deepcopy(center_model)
            client_model = client_model.to(device)
            optimizer = optim.SGD(client_model.parameters(), lr=lr)
            for j in range(local_sgd):
                inputs, labels = train_loader_list[idx].get_next_batch()
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = client_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                record_utils.record_train(loss.item(), outputs, labels)
            new_param = model_utils.get_params_tensor(optimizer, mod_len)
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
        log_util.log(log_file,
                     f"the {i} communication round aggregate finished! current time {round(time.time() - first_agg_time, 3)}s")

        # 每test_time个通信轮进行一次测试
        if i % test_time == test_time - 1:
            with torch.no_grad():
                test_model = copy.deepcopy(center_model)
                test_model = test_model.to(device)
                for data in test_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = test_model(inputs)
                    record_utils.record_test(outputs, labels)

            record_utils.print_test_accuracy(i // test_time)

    log_util.log(log_file, f"训练结束，耗时：{round(time.time() - train_start_time, 3)}s")
