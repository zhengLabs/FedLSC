# -*- coding: utf-8 -*-
# @Time    : 2022 09
# @Author  : yicao
# 对于模型的各种工具类
import csv
import os

import numpy
import torch
import numpy as np

SamplingProportion = 1


# 从优化器中获取梯度
def get_grads_from_optim(optim) -> list:
    grads = []
    for group in optim.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            grads.append(p.grad.data.view(-1))
    return torch.cat(grads).cpu().numpy().tolist()


# 从优化器中获取梯度GPU
def get_grads_from_optim_gpu(optim, mod_len) -> torch.Tensor:
    res_grads = torch.empty(mod_len)
    start = 0
    for group in optim.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad.data.view(-1)
            end = start + len(grad)
            res_grads[start: end] = grad
            start = end
    return res_grads


# 将梯度放回优化器
def put_grads_to_optim_gpu(optim, grads):
    start = 0
    for p in optim.param_groups[0]["params"]:
        if p is None:
            continue
        size = p.data.view(-1).size(0)
        p.grad.data = grads[start: start + size].view(p.grad.shape)
        start += size


# 从优化器中获取梯度的numpy数组
def get_grads_numpy(optim: torch.optim.Optimizer, mod_len: int) -> np.ndarray:
    grads_numpy = np.empty(mod_len)
    start = 0
    for param in optim.param_groups[0]['params']:
        if param.grad is None:
            continue
        grad = param.grad.data.view(-1).cpu().numpy()
        end = start + len(grad)
        grads_numpy[start:end] = grad
        start = end
    return grads_numpy


def get_params_numpy(optim: torch.optim.Optimizer, mod_len: int) -> np.ndarray:
    params_numpy = np.empty(mod_len)
    start = 0
    for param in optim.param_groups[0]['params']:
        param = param.data.view(-1).cpu().numpy()
        end = start + len(param)
        params_numpy[start:end] = param
        start = end
    return params_numpy


# 从优化器中获取梯度的tensor数组
def get_grads_tensor(optim: torch.optim.Optimizer, mod_len: int) -> torch.Tensor:
    grads_tensor = torch.empty(mod_len)
    start = 0
    for param in optim.param_groups[0]['params']:
        if param.grad is None:
            continue
        grad = param.grad.data.view(-1).cpu()
        end = start + len(grad)
        grads_tensor[start:end] = grad
        start = end
    return grads_tensor


def get_params_tensor(optim: torch.optim.Optimizer, mod_len: int) -> torch.Tensor:
    params_tensor = torch.empty(mod_len)
    start = 0
    for param in optim.param_groups[0]['params']:
        if param.grad is None:
            continue
        params = param.data.view(-1).cpu()
        end = start + len(params)
        params_tensor[start:end] = params
        start = end
    return params_tensor


def get_params_tensor_gpu(optim: torch.optim.Optimizer, mod_len: int) -> torch.Tensor:
    params_tensor = torch.empty(mod_len)
    start = 0
    for param in optim.param_groups[0]['params']:
        if param.grad is None:
            continue
        params = param.data.view(-1)
        end = start + len(params)
        params_tensor[start:end] = params
        start = end
    return params_tensor


def params2mod(params: torch.Tensor, mod):
    start = 0
    for p in mod.parameters():
        end = start + len(p.data.view(-1))
        p.data = params[start:end].reshape_as(p.data).clone().detach()
        start = end


# 从优化器中获取梯度2
def get_grads_from_optim2(optim, grads):
    start = 0
    for param in optim.param_groups[0]['params']:
        if param.grad is None:
            continue
        grads_tensor = param.grad.data.view(-1)
        end = start + len(grads_tensor)
        grads[start:end] = grads_tensor.cpu().numpy()
        start = end
    return grads


# 从优化器中获取模型
def get_params_from_optim(optim, params):
    start = 0
    for param in optim.param_groups[0]['params']:
        if param.data is None:
            continue
        param_tensor = param.data.view(-1)
        end = start + len(param_tensor)
        params[start:end] = param_tensor.cpu().numpy()
        start = end
    return params


def add_grads_to_mod(grads: torch.Tensor, mod):
    start = 0
    for p in mod.parameters():
        if p.data is None:
            continue
        end = start + len(p.data.view(-1))
        x = grads[start: end].reshape_as(p.data)
        p.grad = x.clone().detach()
        start = end
    return mod


def get_params_len_from_mod(model):
    length = 0
    for p in model.parameters():
        length += len(p.data.view(-1))
    return length


def get_mod_len_from_optim_except_bias(optim: torch.optim.Optimizer) -> int:
    mod_len = 0
    # 如果模型规律不是奇偶，则可用下面参数判断len(optimizer_center.param_groups[0]['params'][idx].size()) == 1
    for param in optim.param_groups[0]['params'][::2]:
        mod_len += len(param.data.view(-1))
    return mod_len


# 将梯度放入模型中
def set_grads(model, gradient: torch.Tensor) -> None:
    start = 0
    for p in model.parameters():
        end = start + len(p.data.view(-1))
        x = gradient[start:end].reshape_as(p.data)
        p.grad = x.clone().detach()
        start = end


# 创建csv文件并写入表头
def create_csv_and_write_header(optim):
    # 先遍历一边，计算k_array的长度
    k_length = 0
    for idx, param in enumerate(optim.param_groups[0]['params']):
        if idx % 2 == 0:
            t_len = int(param.data.view(-1).size(0) * SamplingProportion)
            k_length += t_len
            with open(os.path.join('results', f"k layer{idx / 2}.csv"), 'w') as f:
                csv_write = csv.writer(f)
                csv_header = np.arange(t_len)
                csv_write.writerow(csv_header)


def create_csv_write_header_2(optim):
    csv_header = ['0', '<50%', '50-100%', '100-150%', '150-200%', '>200%']
    for idx, param in enumerate(optim.param_groups[0]['params']):
        if param.grad is None:
            continue
        with open(os.path.join('results', f"classify k layer{idx}.csv"), 'w') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(csv_header)


# 计算梯度与参数的比值k
def calc_k_10_per(optim):
    # 先遍历一边，计算k_array的长度
    k_length = 0
    for idx, param in enumerate(optim.param_groups[0]['params']):
        if idx % 2 == 0:
            k_length += int(param.data.view(-1).size(0) * SamplingProportion)

    k_array = np.empty(k_length)
    k_start = 0
    for idx, param in enumerate(optim.param_groups[0]['params']):
        # 只观察权重weight，不考虑偏移bias
        if idx % 2 == 0:
            params_array = param.data.view(-1).cpu().numpy()
            grads_array = param.grad.data.view(-1).cpu().numpy()
            temp_array = abs(grads_array / params_array)
            np.random.shuffle(temp_array)
            temp_len = int(temp_array.size * SamplingProportion)
            k_end = k_start + temp_len
            k_array[k_start:k_end] = temp_array[:temp_len]
            with open(os.path.join('results', f"k layer{idx / 2}.csv"), 'a+') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(k_array[k_start:k_end])
            k_start = k_end

    return k_array


# 计算梯度与参数的比值k
def calc_k_classify(optim):
    for idx, param in enumerate(optim.param_groups[0]['params']):
        # 只观察权重weight，不考虑偏移bias
        if param.grad is None:
            continue
        params_array = param.data.view(-1).cpu().numpy()
        grads_array = param.grad.data.view(-1).cpu().numpy()
        temp_array = abs(grads_array / params_array)
        np.random.shuffle(temp_array)
        temp_len = int(temp_array.size * SamplingProportion)
        temp_array = temp_array[:temp_len]
        sec_1 = np.sum(temp_array == 0.0)
        sec_2 = np.sum(temp_array <= 0.5)
        sec_3 = np.sum(temp_array <= 1.0)
        sec_4 = np.sum(temp_array <= 1.5)
        sec_5 = np.sum(temp_array <= 2.0)
        sec_6 = np.sum(temp_array > 2.0)

        with open(os.path.join('results', f"classify k layer{idx}.csv"), 'a+') as f:
            csv_write = csv.writer(f)
            csv_write.writerow([sec_1, sec_2 - sec_1, sec_3 - sec_2, sec_4 - sec_3, sec_5 - sec_4, sec_6])


def clac_k(optim: torch.optim.Optimizer, mod_length: int) -> torch.Tensor:
    """
    根据优化器，返回k（梯度：权值）tensor
    :param optim:
    :param mod_length:
    :return:
    """
    start = 0
    k_tensor = torch.empty(mod_length)
    for param in optim.param_groups[0]['params'][::2]:
        params = param.data.view(-1)
        grads = param.grad.data.view(-1)
        end = start + len(params)
        k_tensor[start:end] = (abs(grads / params)).cpu()
        start = end
    return k_tensor


def sparse_aggregate_params(new_params, last_param, bitmaps):
    params_sum = new_params.sum(0)
    bitmap_sum = bitmaps.sum(0)
    # print(f"bitmap_sum：{round(torch.count_nonzero(bitmap_sum).item() / 121182, 4)}")
    # print(f"params_sum：{round(torch.count_nonzero(params_sum).item() / 121182, 4)}")
    # bitmap_sum = torch.where(bitmap_sum == torch.tensor(0.0), torch.tensor(1.0), bitmap_sum)
    # result = params_sum / bitmap_sum
    # result = torch.where(result == 0, last_param, result)
    result = torch.where(bitmap_sum == torch.tensor(0.0), last_param, params_sum / bitmap_sum)
    # print(f"result：{round(torch.count_nonzero(result).item() / 121182, 4)}")
    return result


def sparse_param2mod(param, mod):
    pass
