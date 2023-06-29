# -*- coding: utf-8 -*-
# @Time    : 2022 09
# @Author  : yicao
import csv
import os
import time
import typing
from typing import Tuple

import numpy as np
import torch

RateK = 0.5


def create_zero_grad(optim: torch.optim.Optimizer):
    """
    创建csv文件，用于记录每层梯度为0的数量
    :param optim:
    :return:
    """
    layer_names = np.arange(int(len(optim.param_groups[0]['params']) / 2))

    with open(os.path.join('results', f"calc zero grad.csv"), 'w') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(layer_names)


def calc_zero_grad(optim: torch.optim.Optimizer):
    """
    记录每层梯度为0的数量
    :param optim:
    :return:
    """
    count_list = []
    for idx, param in enumerate(optim.param_groups[0]['params']):
        # 只观察权重weight，不考虑偏移bias
        if idx % 2 == 0:
            grads_tensor = param.grad.data.view(-1).cpu()
            count = (grads_tensor == 0).sum().item()
            count_list.append(count)

    with open(os.path.join('results', f"calc zero grad.csv"), 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(count_list)


def rate_k_grads(optim: torch.optim.Optimizer, mod_length: int) -> Tuple[np.ndarray, np.ndarray]:
    s_time = time.time()
    k_numpy = np.empty(mod_length)
    grads_numpy = np.empty(mod_length)
    start = 0
    for param in optim.param_groups[0]['params']:
        if param.grad is None:
            continue
        params = param.data.view(-1).cpu().numpy()
        grads = param.grad.data.view(-1).cpu().numpy()
        end = start + len(params)
        k_numpy[start:end] = abs(grads/params)
        grads_numpy[start:end] = grads
        start = end
    # print(f"收集k用时{round(time.time() - s_time, 4)}")
    s_time = time.time()
    indices = np.where(k_numpy > RateK)[0]
    # print(f"计算位图用时{round(time.time() - s_time, 4)}")
    # print(indices.size / mod_length)
    return indices, grads_numpy


def rate_k_grads2(optim: torch.optim.Optimizer, mod_length: int) -> Tuple[np.ndarray, np.ndarray]:
    # s_time = time.time()
    k_numpy = np.empty(mod_length)
    grads_numpy = np.empty(mod_length)
    start = 0
    for param in optim.param_groups[0]['params']:
        if param.grad is None:
            continue
        params = param.data.view(-1)
        grads = param.grad.data.view(-1)
        end = start + len(params)
        k_numpy[start:end] = (abs(grads/params)).cpu().numpy()
        grads_numpy[start:end] = grads.cpu().numpy()
        start = end
    # print(f"收集k用时{round(time.time() - s_time, 4)}")
    # s_time = time.time()
    bitmaps = np.where(k_numpy > RateK, 1, 0)  # 计算位图的开销相比于计算坐标更高，但是位图方便聚合
    grads_numpy *= bitmaps
    # print(f"计算位图用时{round(time.time() - s_time, 4)}")
    # print(bitmaps.size / mod_length)
    return bitmaps, grads_numpy


def rate_k_aggregate(grad_list: typing.List, mod_length: int):
    grads_numpy = np.zeros(mod_length)
    bitmap_numpy = np.zeros(mod_length)
    for bitmap, grads in grad_list:
        grads_numpy += grads
        bitmap_numpy += bitmap
    bitmap_numpy = np.where(bitmap_numpy == 0, 1, bitmap_numpy)
    return grads_numpy / bitmap_numpy
