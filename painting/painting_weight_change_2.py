# %%
import csv
import sys

import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.rcParams['font.sans-serif'] = ['Times New Roman']

# 找到全局最优 10  20 41
# 不断找到最优 15  34
filename3 = "painting/csv/freezing_weight.csv"
colors = ['#FD6D5A', '#FEB40B', '#6DC354', '#994487', '#518CD8', '#443295']

df2 = pd.read_csv(filename3)

# 33 16187
# %%
# s = list(range(40, 45))
df2_v = df2.iloc[:, 41].values.T

# %%
# 绘制大图
# plt.plot(df2_v, linewidth=0.5, color=colors[3], label="spf")
# plt.show()

# zoom_x_start = 200
# zoom_x_end = 300
zoom_x_start = 650
zoom_x_end = 750
# zoom_y_start = -0.05
# zoom_y_end = 0.05
plt.xlim(zoom_x_start, zoom_x_end)
# plt.ylim(zoom_y_start, zoom_y_end)
# 绘制局部放大的图像
plt.plot(df2_v, 'r')

# 添加图例
# plt.legend()

plt.yticks(fontproperties='Times New Roman', size=15)  # 设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=15)
# plt.xlabel('batch', fontsize=18, fontproperties='Times New Roman')
# plt.ylabel('value', fontsize=18, fontproperties='Times New Roman')
# 显示图形
plt.savefig(f"A_Final/smooth-pic/png/weight_change_2.png", dpi=500, bbox_inches='tight')
plt.show()


