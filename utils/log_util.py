# -*- coding: utf-8 -*-
# @Time    : 2023 02
# @Author  : yicao
import os.path
import time


def create_log(file_name):
    day = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    t = time.strftime('%H-%M-%S', time.localtime(time.time()))
    path = os.path.join("logs", day)
    if not os.path.exists(path):
        os.makedirs(path)
    file = os.path.join(path, t + " " + file_name + ".txt")
    with open(file, 'w') as f:
        f.write(file)
        f.write("\r\n")
    print("成功创建文件：" + file)
    return file


def log(file, line):
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    lines = t + " " + line
    with open(file, 'a+') as f:
        f.write(lines)
        f.write("\r\n")
    print(lines)
