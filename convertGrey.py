# -*- coding: utf-8 -*-
# @Time    : 2023 04
# @Author  : yicao
from PIL import Image

# 打开图像
fileName = "SPFS-Alex-in-CIFAR-100.png"
image = Image.open(f'A_Final/smooth-pic/png/{fileName}')

# 转换为黑白
image = image.convert('L')

# 保存转换后的图像
image.save(f"A_Final/smooth-pic/grey/{fileName}")
