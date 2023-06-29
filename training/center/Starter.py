# -*- coding: utf-8 -*-
# @Time    : 2023 05
# @Author  : yicao

import os

import torch

from training.center.CenterSGD import CenterSGD
from training.center.Center import CenterSGDM, CenterAdam, CenterRMSprop
from training.center.CenterSPF import SPF
from utils import public_utils, log_util


def rmsProp(data_name, data_num, mod_name, cuda, lr=0.005, seed=50):
    csv_path = os.path.join('./A-Result-Center', data_name, mod_name, 'CenterRMSprop')
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    file_name = f"CenterRMSprop lr={lr} seed={seed}"
    training = CenterRMSprop(dataset=public_utils.get_data_set(data_name),
                             device=(
                                 torch.device(f'cuda:{cuda}') if torch.cuda.is_available() else torch.device('cpu')),
                             model=public_utils.get_net(mod_name, data_num),
                             log_file=log_util.create_log(f"{file_name} {data_name} {mod_name}"),
                             csv_name=os.path.join(csv_path, file_name),
                             batch_size=100,
                             lr=lr,
                             )
    training.start_training(50)


def sgd(data_name, data_num, mod_name, cuda, lr=0.01, seed=50):
    csv_path = os.path.join('./A-Result-Center', data_name, mod_name, 'CenterSGD')
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    file_name = f"CenterSGD lr={lr} seed={seed}"
    training = CenterSGD(dataset=public_utils.get_data_set(data_name),
                         device=(
                             torch.device(f'cuda:{cuda}') if torch.cuda.is_available() else torch.device('cpu')),
                         model=public_utils.get_net(mod_name, data_num),
                         log_file=log_util.create_log(f"{file_name} {data_name} {mod_name}"),
                         csv_name=os.path.join(csv_path, file_name),
                         batch_size=100,
                         lr=lr,
                         )
    training.start_training(50)


def sgdm(data_name, data_num, mod_name, cuda, lr=0.01, seed=50):
    csv_path = os.path.join('./A-Result-Center', data_name, mod_name, 'CenterSGDM')
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    file_name = f"CenterSGDM lr={lr} seed={seed}"
    training = CenterSGDM(dataset=public_utils.get_data_set(data_name),
                          device=(
                              torch.device(f'cuda:{cuda}') if torch.cuda.is_available() else torch.device('cpu')),
                          model=public_utils.get_net(mod_name, data_num),
                          log_file=log_util.create_log(f"{file_name} {data_name} {mod_name}"),
                          csv_name=os.path.join(csv_path, file_name),
                          batch_size=100,
                          lr=lr,
                          )
    training.start_training(50)


def adam(data_name, data_num, mod_name, cuda, lr=0.001, seed=50):
    csv_path = os.path.join('./A-Result-Center', data_name, mod_name, 'CenterAdam')
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    file_name = f"CenterAdam lr={lr} seed={seed}"
    training = CenterAdam(dataset=public_utils.get_data_set(data_name),
                          device=(
                              torch.device(f'cuda:{cuda}') if torch.cuda.is_available() else torch.device('cpu')),
                          model=public_utils.get_net(mod_name, data_num),
                          log_file=log_util.create_log(f"{file_name} {data_name} {mod_name}"),
                          csv_name=os.path.join(csv_path, file_name),
                          batch_size=100,
                          lr=lr,
                          )
    training.start_training(50)


def spf(data_name, data_num, mod_name, cuda, check_round, lr=0.01, seed=50):
    csv_path = os.path.join('./A-Result-Center', data_name, mod_name, 'CenterSPF')
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    file_name = f"CenterSPF check={check_round} lr={lr} seed={seed}"
    training = SPF(dataset=public_utils.get_data_set(data_name),
                   device=(
                       torch.device(f'cuda:{cuda}') if torch.cuda.is_available() else torch.device('cpu')),
                   model=public_utils.get_net(mod_name, data_num),
                   log_file=log_util.create_log(f"{file_name} {data_name} {mod_name}"),
                   csv_name=os.path.join(csv_path, file_name),
                   batch_size=100,
                   check_round=check_round,
                   lr=lr,
                   )
    training.start_training(50)
