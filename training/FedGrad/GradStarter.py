# -*- coding: utf-8 -*-
# @Time    : 2023 06
# @Author  : yicao
import os

from torch.utils.data import DataLoader, Subset

import training.FedGrad.AVG
from training.FedGrad.AVG import *
from utils import public_utils, log_util
from utils.train_loader_distribute import TrainLoader


class GradStarter:
    def __init__(self, cuda, data_name, data_num, mod_name, seed, batch_size):
        self.batch_size = batch_size
        self.device = (torch.device(f'cuda:{cuda}') if torch.cuda.is_available() else torch.device('cpu'))
        self.data_set = public_utils.get_data_set(data_name)
        self.data_name = data_name
        self.data_num = data_num
        self.mod_name = mod_name
        self.seed = seed

    def get_avg_server(self, lr=0.01):
        _, test_set = self.data_set
        csv_path = os.path.join('./A-Result-Center', self.data_name, self.mod_name, 'AVG')
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        file_name = f'AVG lr={lr} seed={self.seed}'
        log_file = log_util.create_log(f"{file_name} {self.data_name} {self.mod_name}")
        server = AVGServer(
            device=self.device,
            test_loader=DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=0),
            model=public_utils.get_net(self.mod_name, self.data_num),
            log_file=log_file,
            csv_name=os.path.join(csv_path, file_name),
            lr=lr,
        )
        return server, log_file

    def get_avg_client_list(self, client_num, log_file, distribution='iid', lr=0.01, local_sgd=10):
        train_set, _ = self.data_set
        csv_path = os.path.join('./A-Result-Center', self.data_name, self.mod_name, 'AVG')
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        file_name = f'AVG lr={lr} seed={self.seed}'
        client_list = []
        record = RecordTrain(log_file=log_file, train_file=os.path.join(csv_path, file_name), print_freq=local_sgd*client_num)
        data_part = public_utils.get_data_part(self.data_name, distribution, train_set.targets, self.seed, client_num)
        for idx in range(client_num):
            sub_set = Subset(train_set, data_part.client_dict[idx])
            client = Client(
                train_loader=TrainLoader(sub_set, batch_size=self.batch_size, shuffle=True),
                log_file=log_file,
                csv_record=record,
                lr=lr,
                device=self.device,
                local_sgd=local_sgd,

            )
            client_list.append(client)
        return client_list

    def start_training(self, server: training.FedGrad.AVG.AVGServer, client_list: list, comm_round):
        public_utils.set_seed(self.seed)
        for i in range(comm_round):
            model = server.download_model()
            grads = torch.empty([len(client_list), server.mod_len])
            for j, client in enumerate(client_list):
                grads[j] = client.local_train(model)
            server.aggregate(grads)
