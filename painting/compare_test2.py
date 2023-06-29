# %%
import numpy as np
import pandas as pd
import torch
import os

from matplotlib import pyplot as plt

colors2 = ['#FD6D5A', '#FEB40B', '#6DC354', '#994487', '#518CD8', '#443295']
line_styles = ['-', '--', '-', '--', '-', '--']
color = ['red', 'blue', '#FEB40B']

n = 5
length = 50
filenames = [
    'results/fed_avg_topk_residual_Cifar10/s=0.01 test.csv',
    'results/fed_avg_topk_residual_Cifar10/s=0.05 test.csv',
    'results/fed_avg_topk_residual_Cifar10/s=0.1 test.csv',
    'results/fed_avg_topk_residual_Cifar10/s=0.5 test.csv',
    'results/fed_avg_topk_residual_Cifar10/s=1.0 test.csv',
]
pic_title = "LeNet-5 in CIFAR-10 TopK"
line_names = [
    's=0.01',
    's=0.05',
    's=0.1',
    's=0.5',
    's=1.0',
]

df = pd.read_csv(filenames[0]).iloc[:length, 1]
for i in range(n-1):
    df = pd.concat([df, pd.read_csv(filenames[1+i]).iloc[:length, 1]], axis=1, join='outer')


df.columns = line_names

# %% 平滑最大值
for i in range(n):
    test_max = 0
    for j in range(length):
        if df.iloc[:, i][j] > test_max:
            test_max = df.iloc[:, i][j]
        else:
            df.iloc[:, i][j] = test_max

# %%
df.plot(linewidth=0.5, color=colors2, style=line_styles)
plt.title(pic_title)
plt.xlabel("epoch")
plt.ylabel("test acc")
plt.show()
