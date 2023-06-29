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

from utils import model_utils, public_utils, log_util
from utils.record_util import RecordAccUtil
from utils.train_loader_distribute import TrainLoader


def train(seed=50, client_num=16, batch_size=100, epoch=200, local_sgd=5, lr=0.1
          , dataset='Cifar10', distribution='iid', class_num=10, mod_name='LeNet', check=0, cuda=0):
    train_start_time = time.time()
    public_utils.set_seed(seed)
    log_name = f"epoch-results {os.path.basename(__file__)[:-3]} {dataset}-{distribution} {mod_name}" \
               f" lr={lr} n={client_num} check={check} seed={seed}"
    csv_dir = os.path.join('./epoch-results', f'{dataset}-{distribution}', f'{mod_name}')
    csv_name = os.path.join(csv_dir, f'{os.path.basename(__file__)[:-3]} lr={lr} n={client_num} check={check} seed={seed}')

    device = (torch.device(f'cuda:{cuda}') if torch.cuda.is_available() else torch.device('cpu'))

    train_set, test_set = public_utils.get_data_set(dataset)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=0)
    train_loader_list = []

    data_part = public_utils.get_data_part(dataset, distribution, train_set.targets, seed, client_num)

    for i in range(client_num):
        sub_set = Subset(train_set, data_part.client_dict[i])
        train_loader_list.append(DataLoader(sub_set, batch_size=batch_size, shuffle=True))
        # train_loader_list.append(TrainLoader(sub_set, batch_size=batch_size, shuffle=True))

    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    log_file = log_util.create_log(f"{log_name}")

    # 初始化全局模型
    center_model = public_utils.get_net(mod_name, class_num)
    criterion = nn.CrossEntropyLoss()
    mod_len = model_utils.get_params_len_from_mod(center_model)
    log_util.log(log_file, f"模型总参数：{mod_len}")
    #  记录客户端传过来的参数
    params = torch.empty([client_num, mod_len])

    sub_set_len = len(train_set) // client_num
    test_time = 3 #
    communication_idx = 0
    best_test = 0
    best_count = 0

    record_utils = RecordAccUtil(batch_size=batch_size, train_number=len(train_set), test_number=len(test_set),
                                 print_batch=local_sgd * client_num, log_file=log_file, csv_name=csv_name)

    while True:
        for idx in range(client_num):
            client_model = copy.deepcopy(center_model)
            client_model = client_model.to(device)
            optimizer = optim.SGD(client_model.parameters(), lr=lr)
            for _ in range(local_sgd):
                for _, data in enumerate(train_loader_list[idx]):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = client_model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    record_utils.record_train(loss.item(), outputs, labels)
            params[idx] = model_utils.get_params_tensor(optimizer, mod_len)

        # 聚合参数，更新模型
        grad_mean = params.mean(0)
        model_utils.params2mod(grad_mean, center_model)

        # 每test_time个通信轮进行一次测试
        if communication_idx % test_time == test_time - 1:
            with torch.no_grad():
                test_model = copy.deepcopy(center_model)
                test_model = test_model.to(device)
                for data in test_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = test_model(inputs)
                    record_utils.record_test(outputs, labels)

            test_acc = record_utils.print_test_accuracy(communication_idx // test_time)
            if test_acc > best_test:
                best_test = test_acc
                best_count = 0
            else:
                best_count += 1
                if best_count == 9:
                    break
        communication_idx += 1

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
