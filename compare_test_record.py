# %%
# 比较集中训练的训练精度和测试精度
import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['Times New Roman']

colors2 = ['#FD6D5A', '#FEB40B', '#6DC354', '#994487', '#518CD8', '#443295']
line_styles = ['-', '-', '-', '--', '-', '--']
color = ['red', 'blue', '#FEB40B']

n = 2
length = 700

pic_title = "LeNet in Cifar10 on Center Train Acc"
filenames = [
    'results/Cifar10-center/LeNet/record_sgd lr=0.1 check=5 train.csv',
    # 'results/Cifar10-center/LeNet/record_sgd lr=0.1 check=5 test.csv',
    'results/Cifar10-center/LeNet/record_spf lr=0.1 check=5 train.csv',
    # 'results/Cifar10-center/LeNet/record_spf lr=0.1 check=5 test.csv',
]
line_names = [
    'SGD train acc',
    # 'SGD test acc',
    'SPF train acc',
    # 'SPF test acc',
]

df = pd.read_csv(filenames[0]).iloc[:length, 2]
for i in range(n - 1):
    df = pd.concat([df, pd.read_csv(filenames[1 + i]).iloc[:length, 2]], axis=1, join='outer')

df.columns = line_names

# 平滑最大值
# for i in range(n):
#     test_max = 0
#     for j in range(length):
#         if df.iloc[:, i][j] > test_max:
#             test_max = df.iloc[:, i][j]
#         else:
#             df.iloc[:, i][j] = test_max

# %
df.plot(linewidth=1, color=color, style=line_styles, fontsize=20)
# plt.title(pic_title)

plt.legend(fontsize=20)
plt.ylim(60, 100)
plt.xlabel("Epoch", fontsize=25)
plt.ylabel("Acc", fontsize=25)

plt.savefig(f"./results/A_Final/picture/test-comp-{pic_title}.svg", bbox_inches='tight')
plt.show()
