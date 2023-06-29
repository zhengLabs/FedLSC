import os
import random

"""
&&  前一个运行成功后，才运行后面一个
||  前一个运行失败后，才运行后面一个
;   前一个运行结束后（无论成功与否），才运行后面一个
&   并行执行，独立的关系
"""
"""
--type参数：FedAVG FedSPF FedSPF2 FedSGD FedAPF 
LSTMAVG LSTMSPF LSTMSGD LSTMAPF 
TopKFedAVG TopKFedSPF 
LSTMFedSPFTopK
"""

env = r"/home/jky/anaconda3/envs/zyq-py38/bin/python"
# seeds = random.sample(range(1, 10000), 5)
seeds = [9484, 5215, 2303, 1602, 1706]
cuda = [0, 0, 3, 2, 1]

for seed in seeds:
    os.system(
        # f"CUDA_VISIBLE_DEVICES={cuda[0]} python A_Start_Traing.py --type sgd --seed {seed}"
        # f"& CUDA_VISIBLE_DEVICES={cuda[1]} python A_Start_Traing.py --type sgdm --seed {seed}"
        # f"& CUDA_VISIBLE_DEVICES={cuda[2]} python A_Start_Traing.py --type rmsprop --seed {seed}"
        # f"& CUDA_VISIBLE_DEVICES={cuda[3]} python A_Start_Traing.py --type adam --seed {seed}"
        f"CUDA_VISIBLE_DEVICES={cuda[0]} python A_Start_Traing.py --type spf --seed {seed} --check 10"
        f"& CUDA_VISIBLE_DEVICES={cuda[4]} python A_Start_Traing.py --type spf --seed {seed} --check 20"
        f"& CUDA_VISIBLE_DEVICES={cuda[2]} python A_Start_Traing.py --type spf --seed {seed} --check 30"
        f"& CUDA_VISIBLE_DEVICES={cuda[3]} python A_Start_Traing.py --type spf --seed {seed} --check 40"
    )

# conv-fed
# os.system(f"CUDA_VISIBLE_DEVICES={cuda[0]} {env} training_cov.py --mod_name Alex --type FedSPF3 --epoch 200 --distribution iid  --seed {seeds[3]} "
#           f"& CUDA_VISIBLE_DEVICES={cuda[1]} {env} training_cov.py --mod_name VGG --type FedSPF3 --epoch 200 --distribution iid  --seed {seeds[3]} "
#           f"& CUDA_VISIBLE_DEVICES={cuda[2]} {env} training_cov.py --mod_name ResNet --type FedSPF3 --epoch 200 --distribution iid  --seed {seeds[3]} ")
# os.system(f"CUDA_VISIBLE_DEVICES={cuda[0]} python training_cov.py --mod_name VGG --type FedSPF3 --epoch 200 --distribution iid  --seed {seeds[3]} ")

# # lstm-topk
# os.system(f"CUDA_VISIBLE_DEVICES={cuda[0]} {env} training_lstm_topk.py --sparse 0.1 --type LSTMFedSPFTopK --epoch 800 --distribution iid  --seed {seeds[0]} "
#         f"& CUDA_VISIBLE_DEVICES={cuda[1]} {env} training_lstm_topk.py --sparse 0.05 --type LSTMFedSPFTopK --epoch 800 --distribution iid  --seed {seeds[0]} "
#         f"& CUDA_VISIBLE_DEVICES={cuda[2]} {env} training_lstm_topk.py --sparse 0.01 --type LSTMFedSPFTopK --epoch 800 --distribution iid  --seed {seeds[0]} ")


# os.system(
#     f"CUDA_VISIBLE_DEVICES={cuda[0]} {env} training_topk.py --sparse 0.1 --type TopKFedAVG --dataset Cifar10 --class_num 10 --mod_name LeNet --client_num 16 --epoch 300 --distribution iid  --seed {seeds[0]} "
#     f"& CUDA_VISIBLE_DEVICES={cuda[1]} {env} training_topk.py --sparse 0.1 --type TopKFedAVG --dataset Cifar10 --class_num 10 --mod_name LeNet --client_num 16 --epoch 300 --distribution iid  --seed {seeds[1]} "
#     f"& CUDA_VISIBLE_DEVICES={cuda[2]} {env} training_topk.py --sparse 0.1 --type TopKFedAVG --dataset Cifar10 --class_num 10 --mod_name LeNet --client_num 16 --epoch 300 --distribution iid  --seed {seeds[2]} "
#     f"& CUDA_VISIBLE_DEVICES={cuda[3]} {env} training_topk.py --sparse 0.1 --type TopKFedAVG --dataset Cifar10 --class_num 10 --mod_name LeNet --client_num 16 --epoch 300 --distribution iid  --seed {seeds[3]} ")
#
# os.system(
#     f"CUDA_VISIBLE_DEVICES={cuda[0]} {env} training_topk.py --sparse 0.05 --type TopKFedAVG --dataset Cifar10 --class_num 10 --mod_name LeNet --client_num 16 --epoch 300 --distribution iid  --seed {seeds[0]} "
#     f"& CUDA_VISIBLE_DEVICES={cuda[1]} {env} training_topk.py --sparse 0.05 --type TopKFedAVG --dataset Cifar10 --class_num 10 --mod_name LeNet --client_num 16 --epoch 300 --distribution iid  --seed {seeds[1]} "
#     f"& CUDA_VISIBLE_DEVICES={cuda[2]} {env} training_topk.py --sparse 0.05 --type TopKFedAVG --dataset Cifar10 --class_num 10 --mod_name LeNet --client_num 16 --epoch 300 --distribution iid  --seed {seeds[2]} "
#     f"& CUDA_VISIBLE_DEVICES={cuda[3]} {env} training_topk.py --sparse 0.05 --type TopKFedAVG --dataset Cifar10 --class_num 10 --mod_name LeNet --client_num 16 --epoch 300 --distribution iid  --seed {seeds[3]} ")
#
# os.system(
#     f"CUDA_VISIBLE_DEVICES={cuda[0]} {env} training_topk.py --sparse 0.01 --type TopKFedAVG --dataset Cifar10 --class_num 10 --mod_name LeNet --client_num 16 --epoch 300 --distribution iid  --seed {seeds[0]} "
#     f"& CUDA_VISIBLE_DEVICES={cuda[1]} {env} training_topk.py --sparse 0.01 --type TopKFedAVG --dataset Cifar10 --class_num 10 --mod_name LeNet --client_num 16 --epoch 300 --distribution iid  --seed {seeds[1]} "
#     f"& CUDA_VISIBLE_DEVICES={cuda[2]} {env} training_topk.py --sparse 0.01 --type TopKFedAVG --dataset Cifar10 --class_num 10 --mod_name LeNet --client_num 16 --epoch 300 --distribution iid  --seed {seeds[2]} "
#     f"& CUDA_VISIBLE_DEVICES={cuda[3]} {env} training_topk.py --sparse 0.01 --type TopKFedAVG --dataset Cifar10 --class_num 10 --mod_name LeNet --client_num 16 --epoch 300 --distribution iid  --seed {seeds[3]} ")
#
# os.system(
#     f"CUDA_VISIBLE_DEVICES={cuda[0]} {env} training_topk.py --sparse 1.0 --type TopKFedAVG --dataset Cifar10 --class_num 10 --mod_name LeNet --client_num 16 --epoch 300 --distribution iid  --seed {seeds[0]} "
#     f"& CUDA_VISIBLE_DEVICES={cuda[1]} {env} training_topk.py --sparse 1.0 --type TopKFedAVG --dataset Cifar10 --class_num 10 --mod_name LeNet --client_num 16 --epoch 300 --distribution iid  --seed {seeds[1]} "
#     f"& CUDA_VISIBLE_DEVICES={cuda[2]} {env} training_topk.py --sparse 1.0 --type TopKFedAVG --dataset Cifar10 --class_num 10 --mod_name LeNet --client_num 16 --epoch 300 --distribution iid  --seed {seeds[2]} "
#     f"& CUDA_VISIBLE_DEVICES={cuda[3]} {env} training_topk.py --sparse 1.0 --type TopKFedAVG --dataset Cifar10 --class_num 10 --mod_name LeNet --client_num 16 --epoch 300 --distribution iid  --seed {seeds[3]} ")

print("训练结束")
