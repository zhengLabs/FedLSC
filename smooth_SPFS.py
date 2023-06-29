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
from scipy import interpolate

from utils.SPFS_time import data


def load_csv(path):
    data_read = pd.read_csv(path)
    ll = data_read.values.tolist()
    return np.array(ll).T


plt.rcParams['font.sans-serif'] = ['Times New Roman']

colors = ['#FD6D5A', '#FEB40B', '#6DC354', '#994487', '#518CD8', '#443295']
line_styles = ['-', '-', '-', '-', '-', '-']
marks = ['*', 'o', '<', 'P', 'X', 'H', 'p', 'v', '^', '>', '8', 's', 'h', 'd', 'D']

idx = 1
pic_title = data[idx]["pic_title"]
filenames = data[idx]["filenames"]
download_time = data[idx]["download_time"]
train_time = data[idx]["train_time"]
upload_time = data[idx]["upload_time"]
y_start = data[idx]["y_start"]
y_end = data[idx]["y_end"]

max_t = 8
for j, v in enumerate(filenames):
    _, y = load_csv(filenames[v])
    x = []
    t_total = 0
    y_max = 0
    for i in range(y.size):
        for _ in range(3):
            dt = download_time[v] * (random.randint(90, 110) * 0.01)
            tt = train_time[v] * (random.randint(90, 110) * 0.01)
            ut = upload_time[v] * (random.randint(90, 110) * 0.01)
            t_total += (dt + tt + ut) / 1000
        x.append(t_total)
        y[i] = y_max = max(y_max, y[i])

    # max_t = max(max_t, t_total)  # todo:可能需要定值
    gap = (max_t - t_total + 1e-20) / (y.size * ((max_t - t_total + 1e-20) / t_total))
    x_add = (np.arange(t_total + 1e-4, max_t, gap)).tolist()
    x = x + x_add
    y = y.tolist()
    y_add = [y[-1]] * len(x_add)
    y = y + ([y[-1]] * len(x_add))
    # x = sorted(x)
    # y = sorted(y)
    # x = np.array(x)
    # y = np.array(y)
    # noinspection PyTupleAssignmentBalance
    tck, u = interpolate.splprep([x, y], s=10, task=0, full_output=0, quiet=0, k=3, t=None)
    fittedParameters = interpolate.splev(u, tck)
    x_new = np.array(fittedParameters[0])
    y_new = np.array(fittedParameters[1])
    max_ = 0
    for idx in range(len(y_new)):
        max_ = max(y_new[idx], max_)
        y_new[idx] = max_
    x_new = sorted(x_new)
    y_new = sorted(y_new)
    markevery = 400
    if j >= 3:
        markevery = 200
    plt.plot(x_new, y_new,
             color=colors[j],
             linestyle=line_styles[j],
             label=v,
             linewidth=1,
             marker=marks[j],
             markevery=markevery,
             )

# plt.title(pic_title)
plt.legend(fontsize=15, loc="lower right")
plt.yticks(fontproperties='Times New Roman', size=15)  # 设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=15)
plt.xlabel('Time/1000s', fontsize=18, fontproperties='Times New Roman')
plt.ylabel('Accuracy/%', fontsize=18, fontproperties='Times New Roman')
plt.ylim(y_start, y_end)

plt.savefig(f"A_Final/smooth-pic/png/english/SPFS-{pic_title}.png", dpi=500, bbox_inches='tight')
plt.show()
# print(x_avg)
