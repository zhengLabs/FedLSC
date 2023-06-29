import torch

from utils import model_utils

"""
    解冻后置原值
"""


class SPFSManager:
    def __init__(self, mod_len):
        self.EMA_a = torch.tensor(0.90)  # 动量因子
        self.Ek = torch.tensor(0.0)
        self.Ek_abs = torch.tensor(0.0)
        self.freezing_bitmap = torch.ones(mod_len)  # 冻结位图，1解冻，0冻结
        self.last_mod_params = None
        self.Ts = 0.05  # 稳定性阈值
        self.mod_len = mod_len
        self.one = torch.tensor(1.0, dtype=torch.float32)

    def calc_pk(self, params):
        g = params - self.last_mod_params
        self.Ek = torch.where(self.freezing_bitmap == self.one, self.EMA_a * self.Ek + (1 - self.EMA_a) * g, self.Ek)
        self.Ek_abs = torch.where(self.freezing_bitmap == self.one,
                                  self.EMA_a * self.Ek_abs + (1 - self.EMA_a) * abs(g), self.Ek_abs)
        pk = torch.where(self.Ek == torch.tensor([0.]), self.Ek, abs(self.Ek) / self.Ek_abs)
        return pk

    def get_params(self, optim: torch.optim.Optimizer):
        # 1. 提取参数
        params = model_utils.get_params_tensor(optim, self.mod_len)
        # 2. 根据位图，将冻结的参数置0
        return params * self.freezing_bitmap

    def check(self, params):
        if self.last_mod_params is None:
            self.last_mod_params = params
            return

        # 1. 计算非冻结参数的稳定性
        pk = self.calc_pk(params)
        self.last_mod_params = params
        # self.record_pk(pk, params)
        # 2. 如果非冻结参数稳定，则冻结，其他一律解冻
        self.freezing_bitmap = torch.where(((self.freezing_bitmap == torch.tensor(1.0)) & (pk <= self.Ts)),
                                           torch.tensor(0.0),
                                           torch.tensor(1.0))
        # self.Ek *= self.freezing_bitmap
        # self.Ek_abs *= self.freezing_bitmap
        self.record_sparse()

    def record_sparse(self):
        print(f"稀疏度：{round((torch.count_nonzero(self.freezing_bitmap) / self.mod_len * 100).item(), 2)}%")


if __name__ == '__main__':
    f_map = torch.tensor([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.])
    params = torch.tensor([1., 2., 3., 4., 5., 1., 3., 5., 7., 9.])
    Ek = params * 0.1
    params1 = torch.where(f_map == 1, params*0.9, params)
    print()
