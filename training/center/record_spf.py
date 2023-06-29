# -*- coding: utf-8 -*-
# @Time    : 2023 03
# @Author  : yicao
import csv
import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from utils import model_utils, public_utils, log_util, SPFSManager
from utils.record_util import RecordAccUtil


def train(seed=50, dataset='Cifar10', class_num=10, mod_name='LeNet', check=5, cuda=0, batch_size=100, epoch=200,
          lr=0.1):
    train_start_time = time.time()
    public_utils.set_seed(seed)
    log_name = f"{os.path.basename(__file__)[:-3]} {dataset}-center {mod_name}" \
               f" lr={lr} check={check}"
    csv_dir = os.path.join('./results', f'{dataset}-center', f'{mod_name}')
    csv_name = os.path.join(csv_dir, f'{os.path.basename(__file__)[:-3]} lr={lr} check={check}')
    device = (torch.device(f'cuda:{cuda}') if torch.cuda.is_available() else torch.device('cpu'))
    train_set, test_set = public_utils.get_data_set(dataset)
    train_loader = DataLoader(train_set, batch_size=100, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=0)

    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    log_file = log_util.create_log(f"{log_name}")

    # 初始化全局模型
    center_model = public_utils.get_net(mod_name, class_num).to(device)
    optimizer = optim.SGD(center_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    mod_len = model_utils.get_params_len_from_mod(center_model)
    log_util.log(log_file, f"模型总参数：{mod_len}")

    # 初始化APFManager
    spfs_manager = SPFSManager.SPFSManager(mod_len)

    record_utils = RecordAccUtil(batch_size=batch_size, train_number=len(train_set), test_number=len(test_set),
                                 print_batch=50, log_file=log_file, csv_name=csv_name)

    # 创建记录权重的csv
    sample_idx = np.random.randint(0, mod_len, 200)
    with open(os.path.join(csv_dir, f'{os.path.basename(__file__)[:-3]} weight.csv'), 'w') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(['iter'] + list(sample_idx))

    for i in range(epoch):
        for idx, data in enumerate(train_loader):
            center_model.train()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = center_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            record_utils.record_train(loss.item(), outputs, labels)
            params = spfs_manager.get_params(optimizer)
            with open(os.path.join(csv_dir, f'{os.path.basename(__file__)[:-3]} weight.csv'), 'a+') as f:
                csv_write = csv.writer(f)
                csv_write.writerow([i*(len(train_loader)) + idx] + list(params[sample_idx].numpy()))

            model_utils.params2mod(params.to(device), center_model)

        params = model_utils.get_params_tensor(optimizer, mod_len)
        with torch.no_grad():
            center_model.eval()
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = center_model(inputs)
                record_utils.record_test(outputs, labels)
        record_utils.print_test_accuracy(i)

        spfs_manager.check(params)
        model_utils.params2mod(params.to(device), center_model)

    log_util.log(log_file, f"训练结束，耗时：{round(time.time() - train_start_time, 3)}s")


if __name__ == '__main__':
    train()
