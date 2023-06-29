# %%
import numpy as np
import pandas as pd
import torch
import os

from matplotlib import pyplot as plt

colors2 = ['#FD6D5A', '#FEB40B', '#6DC354', '#994487', '#518CD8', '#443295']
line_styles = ['-', '--', '-', '--', '-', '--']
color = ['red', 'blue', '#FEB40B']

n = 4
length = 200

# pic_title = "LeNet in Cifar10"
# filenames = [
#     'results/Cifar10-iid/LeNet/FedAVG lr=0.1 n=16 check=4 test.csv',
#     'results/Cifar10-iid/LeNet/FedSGD lr=0.1 n=16 check=4 test.csv',
#     'results/Cifar10-iid/LeNet/FedAPF lr=0.1 n=16 check=4 test.csv',
#     'results/Cifar10-iid/LeNet/FedSPF lr=0.1 n=16 check=4 test.csv',
# ]
# pic_title = "Alex in Cifar100"
# filenames = [
#     'results/Cifar100-iid/Alex/FedAVG lr=0.1 n=16 check=4 test.csv',
#     'results/Cifar100-iid/Alex/FedSGD lr=0.1 n=16 check=4 test.csv',
#     'results/Cifar100-iid/Alex/FedAPF lr=0.1 n=16 check=4 test.csv',
#     'results/Cifar100-iid/Alex/FedSPF lr=0.1 n=16 check=4 test.csv',
# ]
# pic_title = "VGG in Cifar10"
# filenames = [
#     'results/Cifar10-iid/VGG/FedAVG lr=0.1 n=16 check=4 test.csv',
#     'results/Cifar10-iid/VGG/FedSGD lr=0.1 n=16 check=4 test.csv',
#     'results/Cifar10-iid/VGG/FedAPF lr=0.1 n=16 check=4 test.csv',
#     'results/Cifar10-iid/VGG/FedSPF lr=0.1 n=16 check=4 test.csv',
# ]]
# pic_title = "ResNet in Cifar10"
# filenames = [
#     'results/Cifar10-iid/ResNet/FedAVG lr=0.1 n=16 check=4 test.csv',
#     'results/Cifar10-iid/ResNet/FedSGD lr=0.1 n=16 check=4 test.csv',
#     'results/Cifar10-iid/ResNet/FedAPF lr=0.1 n=16 check=4 test.csv',
#     'results/Cifar10-iid/ResNet/FedSPF lr=0.1 n=16 check=4 test.csv',
# ]
# pic_title = "ResNet in Cifar100"
# filenames = [
#     'results/Cifar100-iid/ResNet/FedAVG lr=0.1 n=16 check=4 test.csv',
#     'results/Cifar100-iid/ResNet/FedSGD lr=0.1 n=16 check=4 test.csv',
#     'results/Cifar100-iid/ResNet/FedAPF lr=0.1 n=16 check=4 test.csv',
#     'results/Cifar100-iid/ResNet/FedSPF lr=0.1 n=16 check=4 test.csv',
# ]
# pic_title = "LSTM in IMDB"
# filenames = [
#     'results/IMDB-iid/lstm/LSTMFedAVG lr=0.5 n=8 check=5 test.csv',
#     'results/IMDB-iid/lstm/LSTMFedSGD lr=0.5 n=8 check=5 test.csv',
#     'results/IMDB-iid/lstm/LSTMFedAPF lr=0.5 n=8 check=5 test.csv',
#     'results/IMDB-iid/lstm/LSTMFedSPF lr=0.5 n=8 check=5 test.csv'
# ]

line_names = [
    'AVG',
    'SGD',
    'APF',
    'SPF',
]

df = pd.read_csv(filenames[0]).iloc[:length, 1]
for i in range(n-1):
    df = pd.concat([df, pd.read_csv(filenames[1+i]).iloc[:length, 1]], axis=1, join='outer')


df.columns = line_names

# % 平滑最大值
# for i in range(n):
#     test_max = 0
#     for j in range(length):
#         if df.iloc[:, i][j] > test_max:
#             test_max = df.iloc[:, i][j]
#         else:
#             df.iloc[:, i][j] = test_max

# %
df.plot(linewidth=1, color=colors2, style=line_styles)
plt.title(pic_title)
# plt.xlim(65,85)
plt.ylim(75,85)
plt.xlabel("epoch")
plt.ylabel("test acc")
# plt.savefig(f"./painting/pic/test-comp-{pic_title}.png", dpi=500)
plt.show()
