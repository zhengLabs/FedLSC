# %%
import numpy as np
import pandas as pd
import torch
import os

from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['Times New Roman']

colors2 = ['#FD6D5A', '#FEB40B', '#6DC354', '#994487', '#518CD8', '#443295']
line_styles = ['-', '-', '-', '-', '--', '-']
color = ['red', 'blue', '#FEB40B']

n = 4
length = 200

pic_title = "LeNet-in-Cifar10"
filenames = [
    'results/A_Final/Cifar10-iid/LeNet/FedAVG lr=0.1 n=16 check=5 seed=9484 test.csv',
    'results/A_Final/Cifar10-iid/LeNet/FedSGD lr=0.1 n=16 check=5 seed=9484 test.csv',
    'results/A_Final/Cifar10-iid/LeNet/FedAPF lr=0.1 n=16 check=5 seed=9484 test.csv',
    'results/A_Final/Cifar10-iid/LeNet/FedSPF lr=0.1 n=16 check=5 seed=5215 test.csv',
]
y_start = 64
y_end = 70

# pic_title = "VGG-in-Cifar10"
# filenames = [
#     'results/A_Final/Cifar10-iid/VGG/FedAVG lr=0.1 n=16 check=5 seed=2303 test.csv',
#     'results/A_Final/Cifar10-iid/VGG/FedSGD lr=0.1 n=16 check=5 seed=5215 test.csv',
#     'results/A_Final/Cifar10-iid/VGG/FedAPF lr=0.1 n=16 check=5 seed=9484 test.csv',
#     'results/A_Final/Cifar10-iid/VGG/FedSPF lr=0.1 n=16 check=5 seed=1706 test.csv',
# ]
# y_start = 78
# y_end = 86

# pic_title = "ResNet-in-Cifar100"
# filenames = [
#     'results/A_Final/Cifar100-iid/ResNet/FedAVG lr=0.1 n=16 check=4 test.csv',
#     'results/A_Final/Cifar100-iid/ResNet/FedSGD lr=0.1 n=16 check=4 test.csv',
#     'results/A_Final/Cifar100-iid/ResNet/FedAPF lr=0.1 n=16 check=4 test.csv',
#     'results/A_Final/Cifar100-iid/ResNet/FedSPF lr=0.1 n=16 check=4 test.csv',
# ]
# y_start = 52
# y_end = 58

# pic_title = "AlexNet-in-Cifar100"
# filenames = [
#     'results/A_Final/Cifar100-iid/Alex/FedAVG lr=0.1 n=16 check=5 seed=5215 test.csv',
#     'results/A_Final/Cifar100-iid/Alex/FedSGD lr=0.1 n=16 check=5 seed=9484 test.csv',
#     'results/A_Final/Cifar100-iid/Alex/FedAPF lr=0.1 n=16 check=5 seed=2303 test.csv',
#     'results/A_Final/Cifar100-iid/Alex/FedSPF lr=0.1 n=16 check=5 seed=1602 test.csv',
# ]
# y_start = 30
# y_end = 44

# pic_title = "No IId LeNet-in-Cifar10"
# filenames = [
#     'results/A_Final/Cifar10-no-iid/LeNet/FedAVG lr=0.1 n=16 check=4 test.csv',
#     'results/A_Final/Cifar10-no-iid/LeNet/FedSGD lr=0.1 n=16 check=4 test.csv',
#     'results/A_Final/Cifar10-no-iid/LeNet/FedAPF lr=0.1 n=16 check=4 test.csv',
#     'results/A_Final/Cifar10-no-iid/LeNet/FedSPF lr=0.1 n=16 check=4 test.csv',
# ]

# pic_title = "No IID VGG-in-Cifar10"
# filenames = [
#     'results/A_Final/Cifar10-no-iid/VGG/FedAVG lr=0.1 n=16 check=5 seed=9484 test.csv',
#     'results/A_Final/Cifar10-no-iid/VGG/FedSGD lr=0.1 n=16 check=5 seed=9484 test.csv',
#     'results/A_Final/Cifar10-no-iid/VGG/FedAPF lr=0.1 n=16 check=5 seed=9484 test.csv',
#     'results/A_Final/Cifar10-no-iid/VGG/FedSPF lr=0.1 n=16 check=5 seed=9484 test.csv',
# ]

# pic_title = "No IId AlexNet-in-Cifar100"
# filenames = [
#     'results/A_Final/Cifar100-no-iid/Alex/FedAVG lr=0.1 n=16 check=4 test.csv',
#     'results/A_Final/Cifar100-no-iid/Alex/FedSGD lr=0.1 n=16 check=4 test.csv',
#     'results/A_Final/Cifar100-no-iid/Alex/FedAPF lr=0.1 n=16 check=4 test.csv',
#     'results/A_Final/Cifar100-no-iid/Alex/FedSPF lr=0.1 n=16 check=4 test.csv',
# ]

# pic_title = "LSTM-in-IMDB"
# filenames = [
#     'results/A_Final/IMDB/LSTM/LSTMFedAVG lr=0.5 n=8 check=5 test.csv',
#     'results/A_Final/IMDB/LSTM/LSTMFedSGD lr=0.5 n=8 check=5 test.csv',
#     'results/A_Final/IMDB/LSTM/LSTMFedAPF lr=0.5 n=8 check=5 test.csv',
#     'results/A_Final/IMDB/LSTM/LSTMFedSPF lr=0.5 n=8 check=5 test.csv',
# ]
# y_start = 78
# y_end = 86


# line_names = [
#     'SGD Train',
#     'SGD Test',
#     'APF Train',
#     'SPF Test',
# ]
line_names = [
    'AVG',
    'SGD',
    'APF',
    'SPF',
]

df = pd.read_csv(filenames[0]).iloc[:length, 1]
for i in range(n - 1):
    df = pd.concat([df, pd.read_csv(filenames[1 + i]).iloc[:length, 1]], axis=1, join='outer')

df.columns = line_names

# 平滑最大值
for i in range(n):
    test_max = 0
    for j in range(length):
        if df.iloc[:, i][j] > test_max:
            test_max = df.iloc[:, i][j]
        else:
            df.iloc[:, i][j] = test_max

# %
df.plot(linewidth=1, color=colors2, style=line_styles, fontsize=20)
plt.legend(fontsize=15)
# plt.title(pic_title)
plt.ylim(y_start, y_end)
plt.xlabel("Test Round", fontsize=25)
plt.ylabel("Accuracy/%", fontsize=25)
plt.yticks(fontproperties='Times New Roman', size=15)  # 设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=15)
plt.savefig(f"results/A_Final/picture/test-comp-{pic_title}.pdf", bbox_inches='tight')
plt.show()
