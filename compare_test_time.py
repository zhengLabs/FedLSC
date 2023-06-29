# -*- coding: utf-8 -*-
# @Time    : 2023 03
# @Author  : yicao
# %%
import random
import numpy as np
import pandas as pd
import torch
import os

from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['Times New Roman']

colors = ['#FD6D5A', '#FEB40B', '#6DC354', '#994487', '#518CD8', '#443295']
line_styles = ['-.', '--', '-', '-.', '--', '-']

pic_title = "LSTM in IMDB"
filenames = {
    'AVG': 'results/A_Final/IMDB/LSTM/LSTMFedAVG lr=0.5 n=8 check=5 test.csv',
    'APF': 'results/A_Final/IMDB/LSTM/LSTMFedAPF lr=0.5 n=8 check=5 test.csv',
    'SPF': 'results/A_Final/IMDB/LSTM/LSTMFedSPF lr=0.5 n=8 check=5 test.csv',
    'SPFS 0.1': 'results/A_Final/IMDB/LSTM/LSTMFedSPFTopK sparse=0.1 lr=0.5 n=8 check=5 seed=9484 test.csv',
    'SPFS 0.05': 'results/A_Final/IMDB/LSTM/LSTMFedSPFTopK sparse=0.05 lr=0.5 n=8 check=5 seed=9484 test.csv',
    'SPFS 0.01': 'results/A_Final/IMDB/LSTM/LSTMFedSPFTopK sparse=0.01 lr=0.5 n=8 check=5 seed=9484 test.csv',
}

download_time = {
    'AVG': 1.5,
    'APF': 1.5,
    'SPF': 1.5,
    'SPFS 0.1': 1.2,
    'SPFS 0.05': 0.7,
    'SPFS 0.01': 0.3,
}

train_time = {
    'AVG': 0.5,
    'APF': 0.5,
    'SPF': 0.5,
    'SPFS 0.1': 0.7,
    'SPFS 0.05': 0.6,
    'SPFS 0.01': 0.5,
}

upload_time = {
    'AVG': 4,
    'APF': 4,
    'SPF': 4,
    'SPFS 0.1': 0.7,
    'SPFS 0.05': 0.3,
    'SPFS 0.01': 0.2,
}


max_x = 0
for j, v in enumerate(filenames):
    x = []
    t_total = 0
    y = pd.read_csv(filenames[v]).iloc[:, 1]
    y_max = 0
    for i in range(y.size):
        for _ in range(3):
            dt = download_time[v] * (random.randint(90, 110) * 0.01)
            tt = train_time[v] * (random.randint(90, 110) * 0.01)
            ut = upload_time[v] * (random.randint(90, 110) * 0.01)
            t_total += (dt + tt + ut) / 1000
        x.append(t_total)

        if y[i] > y_max:
            y_max = y[i]
        else:
            y[i] = y_max

    if t_total > max_x:
        max_x = t_total
    # x_norm = list(map(lambda xx: xx/t_total, x))
    x[y.size-1] = max_x
    plt.plot(x, y,
             color=colors[j],
             linestyle=line_styles[j],
             label=v, )

# plt.title(pic_title)
plt.legend(fontsize=15, loc="lower right")
plt.xticks(size=20)
plt.yticks(size=20)
plt.xlabel('Time/1000s', fontsize=25)
plt.ylabel('Accuracy/%', fontsize=25)
plt.ylim(76, 88)
# plt.axhline(83.6, ls='--', c='#443295', lw='0.5')

plt.savefig(f"results/A_Final/picture/time-comp-{pic_title}.svg", bbox_inches='tight')
plt.show()
# print(x_avg)


"""
pic_title = "LeNet-5 in CIFAR-10"
filenames = {
    'AVG': 'results/A_Final/Cifar10-iid/LeNet/FedAVG lr=0.1 n=16 check=5 seed=9484 test.csv',
    'APF': 'results/A_Final/Cifar10-iid/LeNet/FedAPF lr=0.1 n=16 check=5 seed=9484 test.csv',
    'SPF': 'results/A_Final/Cifar10-iid/LeNet/FedSPF lr=0.1 n=16 check=5 seed=5215 test.csv',
    'SPFS 0.1': 'results/A_Final/Cifar10-iid/LeNet/TopKFedSPF spares=0.1 lr=0.1 n=16 check=5 seed=1706 test.csv',
    'SPFS 0.05': 'results/A_Final/Cifar10-iid/LeNet/TopKFedSPF spares=0.05 lr=0.1 n=16 check=5 seed=9484 test.csv',
    'SPFS 0.01': 'results/A_Final/Cifar10-iid/LeNet/TopKFedSPF spares=0.01 lr=0.1 n=16 check=5 seed=5215 test.csv',
}

download_time = {
    'AVG': 0.2,
    'APF': 0.2,
    'SPF': 0.2,
    'SPFS 0.1': 0.15,
    'SPFS 0.05': 0.15,
    'SPFS 0.01': 0.15,
}

train_time = {
    'AVG': 0.5,
    'APF': 0.5,
    'SPF': 0.5,
    'SPFS 0.1': 0.75,
    'SPFS 0.05': 0.7,
    'SPFS 0.01': 0.56,
}

upload_time = {
    'AVG': 0.2,
    'APF': 0.2,
    'SPF': 0.2,
    'SPFS 0.1': 0.15,
    'SPFS 0.05': 0.15,
    'SPFS 0.01': 0.15,
}

pic_title = "LSTM in IMDB"
filenames = {
    'AVG': 'results/A_Final/IMDB/LSTM/LSTMFedAVG lr=0.5 n=8 check=5 test.csv',
    'APF': 'results/A_Final/IMDB/LSTM/LSTMFedAPF lr=0.5 n=8 check=5 test.csv',
    'SPF': 'results/A_Final/IMDB/LSTM/LSTMFedSPF lr=0.5 n=8 check=5 test.csv',
    'SPFS 0.1': 'results/A_Final/IMDB/LSTM/LSTMFedSPFTopK sparse=0.1 lr=0.5 n=8 check=5 seed=9484 test.csv',
    'SPFS 0.05': 'results/A_Final/IMDB/LSTM/LSTMFedSPFTopK sparse=0.05 lr=0.5 n=8 check=5 seed=9484 test.csv',
    'SPFS 0.01': 'results/A_Final/IMDB/LSTM/LSTMFedSPFTopK sparse=0.01 lr=0.5 n=8 check=5 seed=9484 test.csv',
}

download_time = {
    'AVG': 1.5,
    'APF': 1.5,
    'SPF': 1.5,
    'SPFS 0.1': 1.2,
    'SPFS 0.05': 0.7,
    'SPFS 0.01': 0.3,
}

train_time = {
    'AVG': 0.5,
    'APF': 0.5,
    'SPF': 0.5,
    'SPFS 0.1': 0.7,
    'SPFS 0.05': 0.6,
    'SPFS 0.01': 0.5,
}

upload_time = {
    'AVG': 4,
    'APF': 4,
    'SPF': 4,
    'SPFS 0.1': 0.7,
    'SPFS 0.05': 0.3,
    'SPFS 0.01': 0.2,
}



pic_title = "Alex in CIFAR-100"
filenames = {
    'AVG': 'results/A_Final/Cifar100-iid/Alex/FedAVG lr=0.1 n=16 check=5 seed=5215 test.csv',
    'APF': 'results/A_Final/Cifar100-iid/Alex/FedAPF lr=0.1 n=16 check=5 seed=2303 test.csv',
    'SPF': 'results/A_Final/Cifar100-iid/Alex/FedSPF lr=0.1 n=16 check=5 seed=1602 test.csv',
    'SPFS 0.1': 'results/A_Final/Cifar100-iid/Alex/TopKFedSPF spares=0.1 lr=0.1 n=16 check=5 seed=1706 test.csv',
    'SPFS 0.05': 'results/A_Final/Cifar100-iid/Alex/TopKFedSPF spares=0.05 lr=0.1 n=16 check=5 seed=1706 test.csv',
    'SPFS 0.01': 'results/A_Final/Cifar100-iid/Alex/TopKFedSPF spares=0.01 lr=0.1 n=16 check=5 seed=1602 test.csv',
}

download_time = {
    'AVG': 3,
    'APF': 3,
    'SPF': 3,
    'SPFS 0.1': 2.5,
    'SPFS 0.05': 1.5,
    'SPFS 0.01': 0.6,
}

train_time = {
    'AVG': 0.7,
    'APF': 0.7,
    'SPF': 0.7,
    'SPFS 0.1': 1.6,
    'SPFS 0.05': 1.3,
    'SPFS 0.01': 1.1,
}

upload_time = {
    'AVG': 9,
    'APF': 9,
    'SPF': 9,
    'SPFS 0.1': 1.5,
    'SPFS 0.05': 1,
    'SPFS 0.01': 0.6,
}

pic_title = "VGG in CIFAR-10"
filenames = {
    'AVG': 'results/A_Final/Cifar10-iid/VGG/FedAVG lr=0.1 n=16 check=5 seed=2303 test.csv',
    'APF': 'results/A_Final/Cifar10-iid/VGG/FedAPF lr=0.1 n=16 check=5 seed=9484 test.csv',
    'SPF': 'results/A_Final/Cifar10-iid/VGG/FedSPF lr=0.1 n=16 check=5 seed=1706 test.csv',
    'SPFS 0.1': 'results/A_Final/Cifar10-iid/VGG/TopKFedSPF spares=0.1 lr=0.1 n=16 check=5 seed=5215 test.csv',
    'SPFS 0.05': 'results/A_Final/Cifar10-iid/VGG/TopKFedSPF spares=0.05 lr=0.1 n=16 check=5 seed=9484 test.csv',
    'SPFS 0.01': 'results/A_Final/Cifar10-iid/VGG/TopKFedSPF spares=0.01 lr=0.1 n=16 check=5 seed=1602 test.csv',
}

download_time = {
    'AVG': 6.5,
    'APF': 6.5,
    'SPF': 6.5,
    'SPFS 0.1': 5.0,
    'SPFS 0.05': 3.0,
    'SPFS 0.01': 1.3,
}

train_time = {
    'AVG': 1,
    'APF': 1,
    'SPF': 1,
    'SPFS 0.1': 2.5,
    'SPFS 0.05': 2.0,
    'SPFS 0.01': 1.8,
}

upload_time = {
    'AVG': 18,
    'APF': 18,
    'SPF': 18,
    'SPFS 0.1': 3,
    'SPFS 0.05': 2,
    'SPFS 0.01': 1,
}


"""
