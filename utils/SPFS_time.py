# -*- coding: utf-8 -*-
# @Time    : 2023 04
# @Author  : yicao

data = [
    {
        "pic_title": "LSTM-in-IMDB",
        "filenames": {
            'AVG': 'results/A_Final/IMDB/LSTM/LSTMFedAVG lr=0.5 n=8 check=5 test.csv',
            'APF': 'results/A_Final/IMDB/LSTM/LSTMFedAPF lr=0.5 n=8 check=5 test.csv',
            'SPF': 'results/A_Final/IMDB/LSTM/LSTMFedSPF lr=0.5 n=8 check=5 test.csv',
            'SPFS s= 0.1': 'results/A_Final/IMDB/LSTM/LSTMFedSPFTopK sparse=0.1 lr=0.5 n=8 check=5 seed=9484 test.csv',
            'SPFS s= 0.05': 'results/A_Final/IMDB/LSTM/LSTMFedSPFTopK sparse=0.05 lr=0.5 n=8 check=5 seed=9484 test.csv',
            'SPFS s= 0.01': 'results/A_Final/IMDB/LSTM/LSTMFedSPFTopK sparse=0.01 lr=0.5 n=8 check=5 seed=9484 test.csv',
        },
        "download_time": {
            'AVG': 1.5,
            'APF': 1.5,
            'SPF': 1.5,
            'SPFS s= 0.1': 1.2,
            'SPFS s= 0.05': 0.7,
            'SPFS s= 0.01': 0.3,
        },
        "train_time": {
            'AVG': 0.5,
            'APF': 0.5,
            'SPF': 0.5,
            'SPFS s= 0.1': 0.7,
            'SPFS s= 0.05': 0.6,
            'SPFS s= 0.01': 0.5,
        },
        "upload_time": {
            'AVG': 4,
            'APF': 4,
            'SPF': 4,
            'SPFS s= 0.1': 0.7,
            'SPFS s= 0.05': 0.3,
            'SPFS s= 0.01': 0.2,
        },
        "y_start": 70,
        "y_end": 88,
    },
    {
        "pic_title": "LeNet-5-in-CIFAR-10",
        "filenames": {
            'AVG': 'results/A_Final/Cifar10-iid/LeNet/FedAVG lr=0.1 n=16 check=5 seed=9484 test.csv',
            'APF': 'results/A_Final/Cifar10-iid/LeNet/FedAPF lr=0.1 n=16 check=5 seed=9484 test.csv',
            'SPF': 'results/A_Final/Cifar10-iid/LeNet/FedSPF lr=0.1 n=16 check=5 seed=5215 test.csv',
            'SPFS s= 0.1': 'results/A_Final/Cifar10-iid/LeNet/TopKFedSPF spares=0.1 lr=0.1 n=16 check=5 seed=1706 test.csv',
            'SPFS s= 0.05': 'results/A_Final/Cifar10-iid/LeNet/TopKFedSPF spares=0.05 lr=0.1 n=16 check=5 seed=9484 test.csv',
            'SPFS s= 0.01': 'results/A_Final/Cifar10-iid/LeNet/TopKFedSPF spares=0.01 lr=0.1 n=16 check=5 seed=5215 test.csv',
        },
        "download_time": {
            'AVG': 0.2,
            'APF': 0.2,
            'SPF': 0.2,
            'SPFS s= 0.1': 0.15,
            'SPFS s= 0.05': 0.15,
            'SPFS s= 0.01': 0.15,
        },
        "train_time": {
            'AVG': 0.5,
            'APF': 0.5,
            'SPF': 0.5,
            'SPFS s= 0.1': 0.75,
            'SPFS s= 0.05': 0.7,
            'SPFS s= 0.01': 0.56,
        },
        "upload_time": {
            'AVG': 0.2,
            'APF': 0.2,
            'SPF': 0.2,
            'SPFS s= 0.1': 0.15,
            'SPFS s= 0.05': 0.15,
            'SPFS s= 0.01': 0.15,
        },
        "y_start": 62,
        "y_end": 70.5,
    },
    {
        "pic_title": "VGG-in-CIFAR-10",
        "filenames": {
            'AVG': 'results/A_Final/Cifar10-iid/VGG/FedAVG lr=0.1 n=16 check=5 seed=2303 test.csv',
            'APF': 'results/A_Final/Cifar10-iid/VGG/FedAPF lr=0.1 n=16 check=5 seed=9484 test.csv',
            'SPF': 'results/A_Final/Cifar10-iid/VGG/FedSPF lr=0.1 n=16 check=5 seed=1706 test.csv',
            'SPFS s= 0.1': 'results/A_Final/Cifar10-iid/VGG/TopKFedSPF spares=0.1 lr=0.1 n=16 check=5 seed=5215 test.csv',
            'SPFS s= 0.05': 'results/A_Final/Cifar10-iid/VGG/TopKFedSPF spares=0.05 lr=0.1 n=16 check=5 seed=9484 test.csv',
            'SPFS s= 0.01': 'results/A_Final/Cifar10-iid/VGG/TopKFedSPF spares=0.01 lr=0.1 n=16 check=5 seed=1602 test.csv',
        },
        "download_time": {
            'AVG': 6.5,
            'APF': 6.5,
            'SPF': 6.5,
            'SPFS s= 0.1': 5.0,
            'SPFS s= 0.05': 3.0,
            'SPFS s= 0.01': 1.3,
        },
        "train_time": {
            'AVG': 1,
            'APF': 1,
            'SPF': 1,
            'SPFS s= 0.1': 2.5,
            'SPFS s= 0.05': 2.0,
            'SPFS s= 0.01': 1.8,
        },
        "upload_time": {
            'AVG': 18,
            'APF': 18,
            'SPF': 18,
            'SPFS s= 0.1': 3,
            'SPFS s= 0.05': 2,
            'SPFS s= 0.01': 1,
        },
        "y_start": 78,
        "y_end": 86,
    },
    {
        "pic_title": "Alex-in-CIFAR-100",
        "filenames": {
            'AVG': 'results/A_Final/Cifar100-iid/Alex/FedAVG lr=0.1 n=16 check=5 seed=5215 test.csv',
            'APF': 'results/A_Final/Cifar100-iid/Alex/FedAPF lr=0.1 n=16 check=5 seed=2303 test.csv',
            'SPF': 'results/A_Final/Cifar100-iid/Alex/FedSPF lr=0.1 n=16 check=5 seed=1602 test.csv',
            'SPFS s= 0.1': 'results/A_Final/Cifar100-iid/Alex/TopKFedSPF spares=0.1 lr=0.1 n=16 check=5 seed=1706 test.csv',
            'SPFS s= 0.05': 'results/A_Final/Cifar100-iid/Alex/TopKFedSPF spares=0.05 lr=0.1 n=16 check=5 seed=1706 test.csv',
            'SPFS s= 0.01': 'results/A_Final/Cifar100-iid/Alex/TopKFedSPF spares=0.01 lr=0.1 n=16 check=5 seed=1602 test.csv',
        },
        "download_time": {
            'AVG': 3,
            'APF': 3,
            'SPF': 3,
            'SPFS s= 0.1': 2.5,
            'SPFS s= 0.05': 1.5,
            'SPFS s= 0.01': 0.6,
        },
        "train_time": {
            'AVG': 0.7,
            'APF': 0.7,
            'SPF': 0.7,
            'SPFS s= 0.1': 1.6,
            'SPFS s= 0.05': 1.3,
            'SPFS s= 0.01': 1.1,
        },
        "upload_time": {
            'AVG': 9,
            'APF': 9,
            'SPF': 9,
            'SPFS s= 0.1': 1.5,
            'SPFS s= 0.05': 1,
            'SPFS s= 0.01': 0.6,
        },
        "y_start": 30,
        "y_end": 44,
    },
]
