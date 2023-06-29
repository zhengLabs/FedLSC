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

w1 = [1., 0.98, 0.93, 0.87, 0.873, 0.867, 0.87, 0.83, 0.80, 0.75,
      0.70, 0.70, 0.705, 0.65, 0.55, 0.553, 0.548, 0.552, 0.55, 0.547]

w2 = [
    0.02, 0.03, 0.08, 0.15, 0.153, 0.152, 0.148, 0.146, 0.17, 0.22,
    0.25, 0.29, 0.38, 0.385, 0.381, 0.379, 0.378, 0.383, 0.381, 0.377
]

# w2 = list(range(11, 21, 1))
# y2_ = [(i*0.04)**3 for i in w2]
# y2 = y2_ + [i + (random.random()-0.5)*0.1 for i in [y2_[-1]]*10]

x = np.arange(0, len(w1), 1)

plt.plot(x, w1, "b", label=r'$w_1$')
plt.plot(x, w2, "r", label=r'$w_2$')

# plt.title('Weight Changes in Training')
plt.xlabel('time', fontsize=25)
plt.ylabel('weight', fontsize=25)

# plt.axvline(4, ls='--', c='y', lw='0.5')
plt.axhline(0.87, ls='--', c='y', lw='0.5')
plt.axhline(0.70, ls='--', c='y', lw='0.5')
plt.axhline(0.55, ls='--', c='y', lw='0.5')
# plt.axvline(9, ls='--', c='g', lw='0.8')
plt.axhline(0.15, ls='--', c='g', lw='0.5')
plt.axhline(0.38, ls='--', c='g', lw='0.5')

plt.axis([1, 19, -0.1, 1.15])
plt.xticks([], [])
plt.yticks([0.87, 0.70, 0.55, 0.15, 0.38]
           , [r'$ w^*_{11} $', r'$ w^*_{12} $', r'$ w^*_{13} $', r'$ w^*_{21} $',
              r'$ w^*_{22} $']
           , size=20)

plt.legend(fontsize=20)
plt.savefig(f"./results/A_Final/picture/sketch-2.svg", bbox_inches='tight')

plt.show()
