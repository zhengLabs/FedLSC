# %%
import csv
import sys

import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt

# plt.rcParams['font.sans-serif'] = ['Times New Roman']

# 找到全局最优 10  20 41
# 不断找到最优 15  34
filename1 = "results/Cifar10-center/LeNet/record_sgd weight.csv"
filename2 = "results/Cifar10-center/LeNet/record_spf weight.csv"
filename3 = "painting/csv/freezing_weight.csv"
colors = ['#FD6D5A', '#FEB40B', '#6DC354', '#994487', '#518CD8', '#443295']

df1 = pd.read_csv(filename1)
df2 = pd.read_csv(filename3)

# 33
# %%
# s = list(range(40, 45))
s = [41]
df1_v = df1.iloc[:35000, s].values.T
df2_v = df2.iloc[:, s].values.T

for i in range(len(s)):
    plt.clf()
    plt.title(f"weight changed of {s[i]}")
    # plt.plot(df1_v[i], linewidth=0.5, color=colors[0], label="sgd")
    plt.plot(df2_v[i], linewidth=0.5, color=colors[3], label="spf")
    plt.legend()
    plt.show()

# %% 保存图片
plt.clf()
plt.figure(figsize=(6, 4))
# plt.title(f"Weight Changed of Parameter in SPF")
plt.xticks(size=20)
plt.yticks(size=20)
plt.xlabel("Time", fontsize=25)
plt.ylabel("Value", fontsize=25)

plt.plot(df2_v[1][:2150], linewidth=0.5, color='r', label="spf")
# plt.savefig(f"./painting/pic/weight_change_spf_41.svg", bbox_inches='tight')
plt.show()
