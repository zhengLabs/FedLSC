# -*- coding: utf-8 -*-
# @Time    : 2022 09
# @Author  : yicao
import os
import random
import argparse

import numpy as np
import torch
from torchvision import transforms
import torchvision
from fedlab.utils.dataset.partition import CIFAR10Partitioner, CIFAR100Partitioner

from models import LeNet5, VGG, MyResNet
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


def get_data_set(set_name):
    train_set, test_set = None, None
    if set_name == 'Fmnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.1307, 0.3081)
        ])
        train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True,
                                                      transform=transform_train)
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True,
                                                     transform=transform_train)
    elif set_name == "Cifar10":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
        ])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_train)
    elif set_name == "Cifar100":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
        ])
        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_train)
    else:
        print('数据集名称有误')

    return train_set, test_set


def get_net(net_name, num_classes):
    if net_name == 'LeNet':
        return LeNet5.LeNet5(num_classes=num_classes)
    elif net_name == 'Alex':
        return VGG.MyAlexNet(num_classes=num_classes)
    elif net_name == 'VGG':
        return VGG.VGGNet16(num_classes=num_classes)
    elif net_name == 'ResNet':
        return MyResNet.Cifar10Res(num_classes)
    else:
        print('网络名称错误')
        return None


def get_data_part(dataset, distribution, targets, seed, client_num):
    if distribution == 'iid':
        if dataset == 'Cifar10':
            return CIFAR10Partitioner(targets, client_num, balance=True, partition="iid", seed=seed)
        elif dataset == 'Cifar100':
            return CIFAR100Partitioner(targets, client_num, balance=True, partition="iid", seed=seed)
        else:
            print("数据集错误")
    elif distribution == 'no-iid-1':
        if dataset == 'Cifar10':
            return CIFAR10Partitioner(targets, client_num, balance=None, partition="dirichlet", dir_alpha=1, seed=seed)
        elif dataset == 'Cifar100':
            return CIFAR100Partitioner(targets, client_num, balance=None, partition="dirichlet", dir_alpha=0.3,
                                       seed=seed)
        else:
            print("数据集错误")
    elif distribution == 'no-iid-2':
        if dataset == 'Cifar10':
            return CIFAR10Partitioner(targets, client_num, balance=None, partition="shards", num_shards=80, seed=seed)
        elif dataset == 'Cifar100':
            return CIFAR100Partitioner(targets, client_num, balance=None, partition="shards", num_shards=80, seed=seed)
        else:
            print("数据集错误")
    else:
        print("分布错误")
    return None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--client_num", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--local_sgd", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--class_num", type=int, default='10')
    parser.add_argument("--dataset", type=str, default='Cifar10')
    parser.add_argument("--distribution", type=str, default='iid')
    parser.add_argument("--mod_name", type=str, default='LeNet')
    parser.add_argument("--check", type=int, default='5')
    parser.add_argument("--cuda", type=int, default='0')

    return parser.parse_args()


def get_args_lstm():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--client_num", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--local_sgd", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--class_num", type=int, default='10')
    parser.add_argument("--dataset", type=str, default='IMDB')
    parser.add_argument("--distribution", type=str, default='iid')
    parser.add_argument("--mod_name", type=str, default='lstm')
    parser.add_argument("--check", type=int, default='5')
    parser.add_argument("--cuda", type=int, default='0')

    return parser.parse_args()


def get_args_lstm_topk():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--client_num", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--local_sgd", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--class_num", type=int, default='10')
    parser.add_argument("--dataset", type=str, default='IMDB')
    parser.add_argument("--distribution", type=str, default='iid')
    parser.add_argument("--mod_name", type=str, default='lstm')
    parser.add_argument("--check", type=int, default='5')
    parser.add_argument("--cuda", type=int, default='0')
    parser.add_argument("--sparse", type=float, default=0.1)

    return parser.parse_args()


def get_args_topk():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--client_num", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--local_sgd", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--class_num", type=int, default='10')
    parser.add_argument("--dataset", type=str, default='Cifar10')
    parser.add_argument("--distribution", type=str, default='iid')
    parser.add_argument("--mod_name", type=str, default='LeNet')
    parser.add_argument("--check", type=int, default='5')
    parser.add_argument("--cuda", type=int, default='0')
    parser.add_argument("--sparse", type=float, default=0.4)

    return parser.parse_args()
