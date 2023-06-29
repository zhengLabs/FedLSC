import csv

import torch
import numpy as np

from utils import model_utils


class APFManager:
    def __init__(self, mod_len):
        self.EMA_a = torch.tensor(0.90)
        self.Ek = torch.tensor(0.0)
        self.Ek_abs = torch.tensor(0.0)
        self.freezing_bitmap = torch.ones(mod_len)
        self.last_mod_params = None
        self.Fs = 10  # 同步频率，多少个本地batch进行一次同步
        self.Fc = 5  # 检查频率，几个同步频率后进行检查
        self.Ts = 0.05
        self.mod_len = mod_len
        self.I_freezing = torch.zeros(mod_len, dtype=torch.int32)  # 参数i的解冻轮次
        self.L_freezing = torch.ones(mod_len, dtype=torch.int32)  # 参数i的冻结周期
        self.K_check = 0  # 检查轮次计数
        self.one = torch.tensor(1.0, dtype=torch.float32)

        # self.sample = np.random.choice(list(range(self.mod_len)), 100, replace=False)
        # print(f"随机抽样下标为:{self.sample}")
        # with open("./results/freezing/pk.csv", 'w') as f:
        #     csv_write = csv.writer(f)
        #     csv_write.writerow(self.sample)
        # with open("./results/freezing/param.csv", 'w') as f:
        #     csv_write = csv.writer(f)
        #     csv_write.writerow(self.sample)
        # with open("./results/freezing/calc_pk.csv", 'w') as f:
        #     csv_write = csv.writer(f)
        #     csv_write.writerow(self.sample)

    def calc_pk(self, params):
        g = params - self.last_mod_params
        self.Ek = torch.where(self.freezing_bitmap == self.one, self.EMA_a * self.Ek + (1 - self.EMA_a) * g, self.Ek)
        self.Ek_abs = torch.where(self.freezing_bitmap == self.one,
                                  self.EMA_a * self.Ek_abs + (1 - self.EMA_a) * abs(g), self.Ek_abs)
        pk = torch.where(self.Ek == torch.tensor([0.]), self.Ek, abs(self.Ek) / self.Ek_abs)
        # with open("./results/freezing/calc_pk.csv", 'a+') as f:
        #     csv_write = csv.writer(f)
        #     csv_write.writerow(self.last_mod_params[self.sample].numpy())
        #     csv_write.writerow(params[self.sample].numpy())
        #     csv_write.writerow(g[self.sample].numpy())
        #     csv_write.writerow(self.Ek[self.sample].numpy())
        #     csv_write.writerow(self.Ek_abs[self.sample].numpy())
        #     csv_write.writerow(pk[self.sample].numpy())
        return pk

    def set_last_mod_params(self, params):
        self.last_mod_params = params

    def get_params(self, optim: torch.optim.Optimizer):
        # 1. 提取参数
        params = model_utils.get_params_tensor(optim, self.mod_len)
        # 2. 根据位图，将冻结的参数回滚
        if self.last_mod_params is not None:
            params = torch.where(self.freezing_bitmap == torch.tensor(0.0), self.last_mod_params, params)
        # 3. 发送已冻结的参数
        return params

    def check(self, params):
        if self.last_mod_params is None:
            self.last_mod_params = params
            return

        # 如果达到检查轮次，检查稳定性
        # 1. 计算非冻结参数的稳定性
        pk = self.calc_pk(params)
        self.last_mod_params = params
        # self.record_pk(pk, params)
        # 2. 如果非冻结参数稳定，则将其冻结周期增加
        self.L_freezing = torch.where(((self.freezing_bitmap == self.one) & (pk < self.Ts)), self.L_freezing + 1,
                                      self.L_freezing // 2)
        # 3. 将冻结周期与检查轮数相加，得到解冻时间
        self.I_freezing = torch.where(self.freezing_bitmap == self.one, self.K_check + self.L_freezing, self.I_freezing)
        # 4. 如果当前轮次未达解冻时间则继续冻结，否则解冻
        self.freezing_bitmap = torch.where(self.K_check < self.I_freezing, torch.tensor(0.0), torch.tensor(1.0))
        self.record_sparse()
        self.K_check += 1

    def record_sparse(self):
        print(f"稀疏度：{round((torch.count_nonzero(self.freezing_bitmap) / self.mod_len * 100).item(), 2)}%")

    # def record_pk(self, pk: torch.Tensor, params):
    #     with open("./results/freezing/pk.csv", 'a+') as f:
    #         csv_write = csv.writer(f)
    #         csv_write.writerow(pk[self.sample].numpy())
    #     with open("./results/freezing/param.csv", 'a+') as f:
    #         csv_write = csv.writer(f)
    #         csv_write.writerow(params[self.sample].numpy())
