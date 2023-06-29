import torch


class FedAvgTopK:
    def __init__(self, mod_len, sparse_rate):
        self.sparse_rate = sparse_rate
        self.mod_len = mod_len
        self.last_params = None

    def get_params(self, new_params):
        if self.last_params is None:
            bitmap = torch.ones(self.mod_len)
        else:
            grad = abs(new_params - self.last_params)
            _, idx = grad.topk(int(self.mod_len * self.sparse_rate))
            bitmap = torch.zeros(self.mod_len)
            bitmap[idx] = torch.tensor(1.0)
        # self.last_params = new_params
        return bitmap

    def get_params_residual(self, new_params, residual):
        if self.last_params is None:
            bitmap = torch.ones(self.mod_len)
        else:
            residual = residual + new_params - self.last_params
            _, idx = abs(residual).topk(int(self.mod_len * self.sparse_rate))
            bitmap = torch.zeros(self.mod_len)
            bitmap[idx] = torch.tensor(1.0)
            # print(f"bitmap：{round(torch.count_nonzero(bitmap).item() / self.mod_len, 2)}")
            new_params *= bitmap
            new_params[idx] = self.last_params[idx] + residual[idx]
            residual[idx] = 0
            # print(f"residual：{round(torch.count_nonzero(residual).item() / self.mod_len, 2)}")
        return bitmap, new_params, residual
