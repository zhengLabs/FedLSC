# -*- coding: utf-8 -*-
# @Time    : 2022 09
# @Author  : yicao
import csv
import os

import math
import numpy as np
import torch

from utils import model_utils


class TopKUtil:
    def __init__(self, mod_len: int, sparse_rate: float = 0.05, record_top_k_value=False,
                 record_top_k_value_csv_name=None, a=1):
        self.record_top_k_value_csv_name = record_top_k_value_csv_name
        self.record_top_k_value = record_top_k_value
        self.sparse_rate = sparse_rate
        self.mod_len = mod_len
        self.top_k_idx = int(sparse_rate * mod_len)
        self.top_k_val = 0.005
        self.a = a  # 增强系数
        self.a_k = np.ones(self.mod_len) * self.a  # 上一次传输距离本轮传输的轮次

        self.rs_global = 0  # 模型实际稀疏度
        self.rs_size = 4  # 稀疏度打印周期
        self.rs_count = 0  # 稀疏度打印计数

        # 循环提取参数索引
        self.loop_size = math.ceil(1 / sparse_rate)
        self.loop_idx = 0

        if self.record_top_k_value & (self.record_top_k_value_csv_name is not None):
            with open(os.path.join('results', self.record_top_k_value_csv_name), 'w') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(['top_k_value'])

    def record_rs(self, bitmap: np.ndarray):
        self.rs_global += np.count_nonzero(bitmap)
        self.rs_count += 1

        # 达到记录数，打印各层稀疏率
        if self.rs_count == self.rs_size:
            self.rs_count = 0
            rs = self.rs_global / self.rs_size / self.mod_len
            d = rs - self.sparse_rate
            self.top_k_val = self.top_k_val * (1 + d)
            # print(f"模型稀疏率为：{round(rs, 3)},  阈值调整后的值为：{round(self.top_k_val, 4)}")
            self.rs_global = 0

    def get_grads_from_optim_use_k_val(self, optim: torch.optim.Optimizer):
        """
        选择大于等于k_val的值进行传输
        :param optim:
        :return:
        """
        grads_numpy = model_utils.get_grads_numpy(optim, self.mod_len)
        bitmap = np.where(abs(grads_numpy) >= self.top_k_val, 1, 0)
        self.record_rs(bitmap)
        return bitmap, grads_numpy * bitmap

    def get_grads_from_optim_use_top_k(self, optim: torch.optim.Optimizer):
        """
        普通topk
        :param optim:
        :return:
        """
        grads_tensor = model_utils.get_grads_tensor(optim, self.mod_len)
        val, idx = abs(grads_tensor).topk(self.top_k_idx)
        bitmap = np.zeros(self.mod_len)
        bitmap[idx] = 1
        if self.record_top_k_value & (self.record_top_k_value_csv_name is not None):
            with open(os.path.join('results', self.record_top_k_value_csv_name), 'a+') as f:
                csv_write = csv.writer(f)
                csv_write.writerow([val[-1].item()])

        return bitmap, (grads_tensor.numpy()) * bitmap

    def get_grads_from_optim_use_top_k_layers(self, optim: torch.optim.Optimizer):
        """
        分层topk
        :param optim:
        :return:
        """
        grads_tensor = torch.empty(self.mod_len)
        bitmap = np.zeros(self.mod_len)
        start = 0
        for param in optim.param_groups[0]['params']:
            if param.grad is None:
                continue
            grad = param.grad.data.view(-1).cpu()
            val, idx = abs(grad).topk(int(self.sparse_rate * len(grad)))
            bitmap[start + idx] = 1
            end = start + len(grad)
            grads_tensor[start:end] = grad
            start = end

        return bitmap, (grads_tensor.numpy()) * bitmap

    def get_grads_from_optim_use_top_k_a(self, optim: torch.optim.Optimizer):
        """
        设置了α的topk
        :param optim:
        :return:
        """
        grads_tensor = model_utils.get_grads_tensor(optim, self.mod_len)
        grads_tensor_a = abs(grads_tensor) * self.a_k
        val, idx = grads_tensor_a.topk(self.top_k_idx)
        bitmap = np.zeros(self.mod_len)
        bitmap[idx] = 1
        self.a_k *= self.a
        self.a_k[idx] = self.a
        if self.record_top_k_value & (self.record_top_k_value_csv_name is not None):
            with open(os.path.join('results', self.record_top_k_value_csv_name), 'a+') as f:
                csv_write = csv.writer(f)
                csv_write.writerow([grads_tensor[idx[-1]].item()])

        return bitmap, (grads_tensor.numpy()) * bitmap

    def get_grad_loop(self, optim: torch.optim.Optimizer):
        """
        循环取参数的方法
        :param optim:
        :return:
        """
        grads_numpy = model_utils.get_grads_numpy(optim, self.mod_len)
        bitmap = np.zeros(self.mod_len)
        bitmap[self.loop_idx::self.loop_size] = 1
        self.loop_idx += 1
        if self.loop_idx == self.loop_size:
            self.loop_idx = 0
        return bitmap, grads_numpy * bitmap

    def get_grad_random(self, optim: torch.optim.Optimizer):
        """
        随机稀疏化方法
        :param optim:
        :return:
        """
        grads_numpy = model_utils.get_grads_numpy(optim, self.mod_len)
        bitmap = np.zeros(self.mod_len)
        bitmap_idx = np.arange(self.mod_len)
        bitmap_idx = np.random.choice(bitmap_idx, self.top_k_idx, replace=False)
        bitmap[bitmap_idx] = 1
        return bitmap, grads_numpy * bitmap

    def get_grad_dryden(self, optim: torch.optim.Optimizer):
        """
        dryden方法，按照正负数分别进行tops
        :param optim:
        :return:
        """
        bitmap = np.zeros(self.mod_len)
        grads_tensor = model_utils.get_grads_tensor(optim, self.mod_len)
        grad_po = torch.where(grads_tensor > torch.tensor(0.0), grads_tensor, torch.tensor(0.0))
        idx_po = grad_po.topk(int(self.sparse_rate * grad_po.size(0))).indices
        grad_ne = torch.where(grads_tensor < torch.tensor(0.0), grads_tensor, torch.tensor(0.0))
        idx_ne = abs(grad_ne).topk(int(self.sparse_rate * grad_ne.size(0))).indices
        bitmap[idx_po] = 1
        bitmap[idx_ne] = 1

        return bitmap, (grads_tensor.numpy()) * bitmap

    def get_grads_top_k_residual(self, optim: torch.optim.Optimizer, residual: torch.Tensor):
        """
        残差topk，累加残差
        :param residual:
        :param optim:
        :return:
        """
        grads_tensor = model_utils.get_grads_tensor(optim, self.mod_len)
        residual += grads_tensor

        val, idx = abs(residual).topk(self.top_k_idx)
        bitmap = np.zeros(self.mod_len)
        bitmap[idx] = 1
        result_numpy = (residual.numpy()) * bitmap
        residual[idx] = 0

        return bitmap, result_numpy

    def get_grads_top_k_residual_momentum(self, optim: torch.optim.Optimizer, residual: torch.Tensor):
        """
        残差topk，动量残差
        :param residual:
        :param optim:
        :return:
        """
        grads_tensor = model_utils.get_grads_tensor(optim, self.mod_len)
        residual = torch.where(residual == 0, grads_tensor,
                               torch.tensor(0.9) * residual + torch.tensor(0.1) * grads_tensor)
        # residual = torch.tensor(0.9) * residual + torch.tensor(0.1) * grads_tensor

        val, idx = abs(residual).topk(self.top_k_idx)
        bitmap = np.zeros(self.mod_len)
        bitmap[idx] = 1
        result_numpy = (residual.numpy()) * bitmap
        residual[idx] = 0

        return bitmap, result_numpy

    @staticmethod
    def aggregate_grad(bitmaps: np.ndarray, grads: np.ndarray):
        bitmap_sum = bitmaps.sum(0)
        grad_sum = grads.sum(0)
        bitmap_sum = np.where(bitmap_sum == 0, 1, bitmap_sum)
        return grad_sum / bitmap_sum
