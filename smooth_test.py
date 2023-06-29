# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

plt.rcParams['font.sans-serif'] = ['Times New Roman']

colors = ['#FD6D5A', '#FEB40B', '#6DC354', '#994487', '#518CD8', '#443295']
line_styles = ['-', '-', '-', '-', '--', '-']
marks = ['*', 'o', '<', 'P', 'X', 'H', 'p', 'v', '^', '>', '8', 's', 'h', 'd', 'D']

pic_title = "LeNet-in-Cifar10"
filenames = [
    'results/A_Final/Cifar10-iid/LeNet/FedAVG lr=0.1 n=16 check=5 seed=9484 test.csv',
    'results/A_Final/Cifar10-iid/LeNet/FedSGD lr=0.1 n=16 check=5 seed=9484 test.csv',
    'results/A_Final/Cifar10-iid/LeNet/FedAPF lr=0.1 n=16 check=5 seed=9484 test.csv',
    'results/A_Final/Cifar10-iid/LeNet/FedSPF lr=0.1 n=16 check=5 seed=5215 test.csv',
]
y_start = 64
y_end = 70

# pic_title = "VGG-in-Cifar10"
# filenames = [
#     'results/A_Final/Cifar10-iid/VGG/FedAVG lr=0.1 n=16 check=5 seed=2303 test.csv',
#     'results/A_Final/Cifar10-iid/VGG/FedSGD lr=0.1 n=16 check=5 seed=5215 test.csv',
#     'results/A_Final/Cifar10-iid/VGG/FedAPF lr=0.1 n=16 check=5 seed=9484 test.csv',
#     'results/A_Final/Cifar10-iid/VGG/FedSPF lr=0.1 n=16 check=5 seed=1706 test.csv',
# ]
# y_start = 78
# y_end = 86

# pic_title = "ResNet-in-Cifar100"
# filenames = [
#     'results/A_Final/Cifar100-iid/ResNet/FedAVG lr=0.1 n=16 check=4 test.csv',
#     'results/A_Final/Cifar100-iid/ResNet/FedSGD lr=0.1 n=16 check=4 test.csv',
#     'results/A_Final/Cifar100-iid/ResNet/FedAPF lr=0.1 n=16 check=4 test.csv',
#     'results/A_Final/Cifar100-iid/ResNet/FedSPF lr=0.1 n=16 check=4 test.csv',
# ]
# y_start = 52
# y_end = 58

# pic_title = "AlexNet-in-Cifar100"
# filenames = [
#     'results/A_Final/Cifar100-iid/Alex/FedAVG lr=0.1 n=16 check=5 seed=5215 test.csv',
#     'results/A_Final/Cifar100-iid/Alex/FedSGD lr=0.1 n=16 check=5 seed=9484 test.csv',
#     'results/A_Final/Cifar100-iid/Alex/FedAPF lr=0.1 n=16 check=5 seed=2303 test.csv',
#     'results/A_Final/Cifar100-iid/Alex/FedSPF lr=0.1 n=16 check=5 seed=1602 test.csv',
# ]
# y_start = 30
# y_end = 44

# pic_title = "LSTM-in-IMDB"
# filenames = [
#     'results/A_Final/IMDB/LSTM/LSTMFedAVG lr=0.5 n=8 check=5 test.csv',
#     'results/A_Final/IMDB/LSTM/LSTMFedSGD lr=0.5 n=8 check=5 test.csv',
#     'results/A_Final/IMDB/LSTM/LSTMFedAPF lr=0.5 n=8 check=5 test.csv',
#     'results/A_Final/IMDB/LSTM/LSTMFedSPF lr=0.5 n=8 check=5 test.csv',
# ]
# y_start = 78
# y_end = 86

# pic_title = "FedAVG-Top-k"
# filenames = [
#     'results/Cifar10-iid/LeNet/TopKFedAVG/spares=1.0 lr=0.1 n=16 check=5 seed=5215 test.csv',
#     'results/Cifar10-iid/LeNet/TopKFedAVG/spares=0.1 lr=0.1 n=16 check=5 seed=9484 test.csv',
#     'results/Cifar10-iid/LeNet/TopKFedAVG/spares=0.01 lr=0.1 n=16 check=5 seed=2303 test.csv',
#     'results/Cifar10-iid/LeNet/TopKFedAVG/spares=0.05 lr=0.1 n=16 check=5 seed=9484 test.csv',
# ]
# y_start = 50
# y_end = 70

# line_names = [
#     's = 1.0',
#     's = 0.1',
#     's = 0.05',
#     's = 0.01',
# ]
line_names = [
    'AVG',
    'SGD',
    'APF',
    'SPF',
]


def load_csv(path):
    data_read = pd.read_csv(path)
    ll = data_read.values.tolist()
    return np.array(ll).T


for i in range(4):
    data = load_csv(filenames[i])
    max_ = 0
    for idx in range(len(data[1])):
        max_ = max(data[1][idx], max_)
        data[1][idx] = max_

    x, y = data
    # noinspection PyTupleAssignmentBalance
    tck, u = interpolate.splprep([x, y], s=10, task=0, full_output=0, quiet=0, k=5, t=None)
    fittedParameters = interpolate.splev(u, tck)
    x_new = np.array(fittedParameters[0])
    y_new = np.array(fittedParameters[1])
    max_ = 0
    for idx in range(len(y_new)):
        max_ = max(y_new[idx], max_)
        y_new[idx] = max_
    x_new = sorted(x_new)
    y_new = sorted(y_new)

    plt.plot(x_new, y_new,
             color=colors[i],
             linestyle=line_styles[i],
             linewidth=1,
             label=line_names[i],
             marker=marks[i],
             markevery=15,
             )

plt.legend(fontsize=15)
plt.xlabel("Test Round", fontsize=18, fontproperties='Times New Roman')
plt.ylabel("Accuracy/%", fontsize=18, fontproperties='Times New Roman')
plt.yticks(fontproperties='Times New Roman', size=15)  # 设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=15)
plt.ylim(y_start, y_end)
plt.savefig(f"A_Final/smooth-pic/png/english/LSC-{pic_title}.png", dpi=500, bbox_inches='tight')
plt.show()
