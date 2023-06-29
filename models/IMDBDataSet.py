import random

import torch
from torch.utils.data import Dataset

class IMDBDataSet(Dataset):
    def __init__(self, text_path, vocab, MAX_LEN):
        self.MAX_LEN = MAX_LEN
        self.vocab = vocab
        file = open(text_path, 'r', encoding='utf-8')
        self.text_with_tag = file.readlines()  # 文本标签与内容
        random.shuffle(self.text_with_tag)
        file.close()

    def __getitem__(self, index): # 重写getitem
        line = self.text_with_tag[index] # 获取一个样本的标签和文本信息
        label = int(line[0]) # 标签信息
        text = line[2:-1]  # 文本信息
        text = self.text_transform(text)
        return text, label

    def __len__(self):
        return len(self.text_with_tag)

    # 根据vocab将句子转为定长MAX_LEN的tensor
    def text_transform(self, sentence):
        sentence_idx = [self.vocab[token] if token in self.vocab.keys() else self.vocab['<UNK>'] for token in tokenizer(sentence)] # 句子分词转为id

        if len(sentence_idx) < self.MAX_LEN:
            for i in range(self.MAX_LEN-len(sentence_idx)): # 对长度不够的句子进行PAD填充
                sentence_idx.append(self.vocab['<PAD>'])

        sentence_idx = sentence_idx[:self.MAX_LEN] # 取前MAX_LEN长度
        return torch.LongTensor(sentence_idx) # 将转为idx的词转为tensor

# 分词方法
def tokenizer(sentence):
    return sentence.split()