# -*- coding: utf-8 -*-
# @Time    : 2023 06
# @Author  : yicao


import random
import numpy as np

# 生成100个随机正整数
random_numbers = np.random.randint(0, 100000, size=100)

# 将随机数保存到文件中
with open('random_numbers.txt', 'w') as f:
    for num in random_numbers:
        f.write(str(num) + '\n')

# 读取文件中的随机数
random_list = []
with open('random_numbers.txt', 'r') as f:
    for line in f:
        random_list.append(int(line.strip()))
        # print(line.strip())
print(random_list)
