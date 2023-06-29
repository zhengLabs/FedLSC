import os
import sys
from torchsummary import summary

sys.path.append("/home/jky/zyq/SPFS")
from models.VGG import VGGNet16

if __name__ == '__main__':
    summary(VGGNet16(), (3, 32, 32))  # Params size (MB): 56.80
