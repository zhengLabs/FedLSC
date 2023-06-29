# -*- coding: utf-8 -*-
# @Time    : 2022 12
# @Author  : yicao
# w1，w2非凸下的参数变化

# %%
import numpy as np
import random
from matplotlib import pyplot as plt

plt.figure(figsize=(8, 6))
plt.rcParams['font.sans-serif'] = ['Times New Roman']

w1 = list(range(20, 10, -2))
y1_ = [(i * 0.1) ** 2 for i in w1]
y1 = y1_ + [i + (random.random() - 0.5) * 0.1 for i in [y1_[-1]] * 15]

w2 = list(range(11, 21, 1))
y2_ = [(i * 0.04) ** 3 for i in w2]
y2 = y2_ + [i + (random.random() - 0.5) * 0.1 for i in [y2_[-1]] * 10]

x = np.arange(0, len(y1), 1)

plt.plot(x, y1, "b", label=r'$w_1$')
plt.plot(x, y2, "r", label=r'$w_2$')

# plt.title('Weight Changes in Training')
plt.xlabel('time', fontsize=25)
plt.ylabel('weight', fontsize=25)

plt.axvline(4, ls='--', c='y', lw='0.8')
plt.axhline(y1_[-1], ls='--', c='y', lw='0.8')
plt.axvline(9, ls='--', c='g', lw='0.8')
plt.axhline(y2_[-1], ls='--', c='g', lw='0.8')

plt.axis([1, 19, -0.5, 4])
plt.xticks([], [])
# plt.xticks([4, 9], ['$ t_1 $', '$ t_2 $'], size=20)
plt.yticks([y1_[-1], y2_[-1]], [r'$ w^*_1 $', '$ w^*_2 $'], size=20)

plt.legend(fontsize=20)

# plt.savefig(f"./results/A_Final/picture/sketch-1.svg", bbox_inches='tight')
plt.show()
