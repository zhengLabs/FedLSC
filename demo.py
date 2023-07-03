# import logging
#
# logging.basicConfi                 level=logging.DEBUG,
# #                     filename='./new.txt',
# #                     filemode='w')
# # logger = logging.getLogger("user")
# # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# # handler = logging.FileHandler('./log_demo.txt', 'w')
# # handler.setFormatter(formatter)
# # handler.setLevel(level=logging.DEBUG)
# # logger.addHandler(handler)
# #
# # logger.info("asasdasdd")
# # logger.debug("123123asdfsdasdds")
# # logger.warning("123123asdasdsdafsds")
#
#
# import copy
# import csv
# import math
# import os
# import time
#
# import numpy as np
# import torch
# from torch import nn, optim
# from torch.utils.data import DataLoader, Subset
#
# from utils import model_utils, public_utils, log_util
# from utils.constant import mod_name
# from utils.record_util import RecordAccUtil
# from utils.train_loader_distribute import TrainLoader
#
# model = public_utils.get_net(mod_name.LE_NET, 10)
# param = []
# for p in model.parameters():
#     # 将张量展平并转换为列表
#     str_list = [str(val) for val in p.data.view(-1).cpu().numpy().round(10)]  # 将列表中的每个值转换为字符串
#     param += str_listg(format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
#

# import matplotlib.pyplot as plt
# import pandas as pd

# # 读取三个文件
# apf_df = pd.read_csv('results/Cifar10-center/LeNet/weight_change_apf.csv')
# avg_df = pd.read_csv('results/Cifar10-center/LeNet/weight_change_avg.csv')
# spf_df = pd.read_csv('results/Cifar10-center/LeNet/weight_change_spf.csv')

# idx_list = range(5, 10)

# for idx in idx_list:
#     # 获取指定参数列的列名
#     col_name = apf_df.columns[idx]

#     # 获取指定参数列的值
#     apf_data = apf_df.iloc[:, idx]
#     avg_data = avg_df.iloc[:, idx]
#     spf_data = spf_df.iloc[:, idx]

#     # 绘制折线图
#     plt.plot(apf_df.iloc[:, 0], apf_data, label='apf')
#     plt.plot(avg_df.iloc[:, 0], avg_data, label='avg')
#     plt.plot(spf_df.iloc[:, 0], spf_data, label='spf')

#     # 添加图例和标签
#     plt.legend()
#     plt.xlabel('Iteration')
#     plt.ylabel(col_name)

#     # 显示图形
#     plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# 读取两个文件
apf_df = pd.read_csv('results/Cifar10-center/LeNet/weight_change_apf.csv')
test_acc_df = pd.read_csv('results/Cifar10-iid/LeNet/FedAPF/lr=0.1 n=16 check=5 seed=9484 test.csv')
# 读取两个文件
spf_df = pd.read_csv('results/Cifar10-center/LeNet/weight_change_spf.csv')
test_acc_df_apf = pd.read_csv('results/Cifar10-iid/LeNet/FedSPF lr=0.1 n=16 check=5 seed=9484 test.csv')

idxes = [58]
# idxes = [39, 51, 58, 72, 78]

# 39 51 58 72 78

# 获取要绘制的参数列
for idx in idxes:
    # 获取指定参数列的列名
    # col_name = apf_df.columns[idx]

    # 获取指定参数列的值
    apf_data = apf_df.iloc[:400, idx]
    test_acc_data = test_acc_df.iloc[:400, 1]

    # 创建图形和轴对象
    fig, ax1 = plt.subplots()

    # 绘制左侧y轴的折线图
    ax1.plot(apf_df.iloc[:400, 0], apf_data, label='Origin')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Parameter Value')

    # 创建右侧y轴的轴对象
    ax2 = ax1.twinx()

    # 绘制右侧y轴的折线图
    ax2.plot(test_acc_df.iloc[:400, 0], test_acc_data, label='Original Accuracy', color='red', linewidth=1)
    ax2.set_ylabel('Test Accuracy')
    ax2.set_ylim(50, 75)

    # 获取指定参数列的列名
    # col_name = spf_df.columns[idx]

    # 获取指定参数列的值
    apf_data = spf_df.iloc[:400, idx]
    test_acc_data = test_acc_df_apf.iloc[:400, 1]

    # 绘制左侧y轴的折线图
    ax1.plot(spf_df.iloc[:400, 0], apf_data, label='Zero')

    # 绘制右侧y轴的折线图
    ax2.plot(test_acc_df_apf.iloc[:400, 0], test_acc_data, label='Zero Accuracy', color='blue', linewidth=1, )

    # 添加图例
    ax1.legend(loc='lower left')
    ax2.legend(loc='lower right')

    # 显示图形
    plt.show()
