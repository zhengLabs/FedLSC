# -*- coding: utf-8 -*-
# @Time    : 2022 12
# @Author  : yicao
import argparse
import copy
import math
import os
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from utils import model_utils, public_utils, log_util, APFManager
from utils.record_util import RecordAccUtil
from utils.train_loader_distribute import TrainLoader


def train(seed=50, client_num=16, batch_size=100, epoch=200, local_sgd=10, lr=0.1
          , dataset='Cifar10', distribution='iid', class_num=10, mod_name='LeNet', check=0, cuda=0):
    train_start_time = time.time()
    public_utils.set_seed(seed)
    log_name = f"{os.path.basename(__file__)[:-3]} {dataset}-{distribution} {mod_name}" \
               f" lr={lr} n={client_num} check={check} seed={seed}"
    csv_dir = os.path.join('./results', f'{dataset}-{distribution}', f'{mod_name}', os.path.basename(__file__)[:-3])
    csv_name = os.path.join(csv_dir, f'lr={lr} n={client_num} check={check} seed={seed}')

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
    log_util.log(log_file, f"模型总参数：{mod_len}")
    # 初始化APFManager
    apf_manager = APFManager.APFManager(mod_len)
    #  记录客户端传过来的参数
    params = torch.empty([client_num, mod_len])

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
                # 回滚
                cur_param = (apf_manager.get_params(optimizer)).to(device)
                model_utils.params2mod(cur_param, client_model)
            params[idx] = apf_manager.get_params(optimizer)

        # 聚合参数，更新模型
        grad_mean = params.mean(0)
        log_util.log(log_file, f"grad_mean稀疏度：{round(100. * torch.count_nonzero(grad_mean).item() / mod_len, 2)}%")
        model_utils.params2mod(grad_mean, center_model)

        if i % check == check - 1:
            apf_manager.check(grad_mean)
            log_util.log(log_file, f"冻结稀疏度：{round(100. * torch.count_nonzero(apf_manager.freezing_bitmap).item() / mod_len, 2)}%")

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


# if __name__ == '__main__':
#     args = public_utils.get_args()
#     train(
#         seed=args.seed,
#         client_num=args.client_num,
#         batch_size=args.batch_size,
#         epoch=args.epoch,
#         local_sgd=args.local_sgd,
#         lr=args.lr,
#         dataset=args.dataset,
#         distribution=args.distribution,
#         class_num=args.class_num,
#         mod_name=args.mod_name,
#         check=args.check,
#         cuda=args.cuda
#     )
