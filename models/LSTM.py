import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class LSTM(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size) # embedding层

        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bidirectional=False)
        self.decoder = nn.Linear(hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        # inputs的形状是（批量大小，词数），因此LSTM需要将序列长度（Seq_len）作为第一维，所以将输入转置后 再提取词特征
        embeddings = self.embedding(inputs.permute(1,0)) # permute(1,0)交换维度
        # LSTM只传入输入embeddings,因此只返回最后一层的隐藏层再各时间步的隐藏状态
        # outputs的形状是（词数，批量大小， 隐藏单元个数）
        outputs, _ = self.encoder(embeddings)
        # 连接初时间步和最终时间步的隐藏状态作为全连接层的输入。形状为(批量大小， 隐藏单元个数)
        encoding = outputs[-1] # 取LSTM最后一层结果
        outs = self.softmax(self.decoder(encoding)) # 输出层为二维概率[a,b]
        return outs


if __name__ == '__main__':
    mod = LSTM(10000, 300, 128, 2)
    summary(mod, (300, ),dtypes=[torch.long])
