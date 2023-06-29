# -*- coding: utf-8 -*-
# @Time    : 2023 05
# @Author  : yicao
import argparse
import os

from training.FedGrad.GradStarter import GradStarter
from training.center import Starter
from utils import public_utils

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--type", type=str)
    # parser.add_argument("--seed", type=int)
    # parser.add_argument("--check", type=int, default=5)
    # args = parser.parse_args()

    data_name = "Cifar10"
    data_num = 10
    mod_name = "LeNet"
    cuda = 0
    seed = 9848
    public_utils.set_seed(seed)
    starter = GradStarter(cuda, data_name, data_num, mod_name, seed, 100)
    server, logfile = starter.get_avg_server(lr=0.1)
    client_list = starter.get_avg_client_list(4, logfile, lr=0.01)
    starter.start_training(server, client_list, 100)


    # csv_path = os.path.join('./A-Result-Center', data_name, mod_name)
    # if not os.path.exists(csv_path):
    #     os.makedirs(csv_path)

    # Starter.sgd(data_name, data_num, mod_name, cuda, lr=0.01)

    # if args.type == 'sgd':
    #     Starter.sgd(data_name, data_num, mod_name, cuda, lr=0.01, seed=seed)
    # elif args.type == 'sgdm':
    #     Starter.sgdm(data_name, data_num, mod_name, cuda, lr=0.01, seed=seed)
    # elif args.type == 'rmsprop':
    #     Starter.rmsProp(data_name, data_num, mod_name, cuda, lr=0.001, seed=seed)
    # elif args.type == 'adam':
    #     Starter.adam(data_name, data_num, mod_name, cuda, lr=0.001, seed=seed)
    # elif args.type == 'spf':
    #     Starter.spf(data_name, data_num, mod_name, cuda, check_round=args.check, lr=0.01, seed=seed)
