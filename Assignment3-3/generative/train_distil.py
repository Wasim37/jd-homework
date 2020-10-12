#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Peixin Lin
Date: 2020-09-29 17:05:16
LastEditTime: 2020-09-30 10:25:44
FilePath: /Assignment3-3/generative/train_distil.py
Desciption: Perform knowledge distillation for compressing the BERT model.
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''
import os
import random
import sys
from functools import partial
import csv

import numpy as np
import pandas as pd
import torch
from textbrewer import DistillationConfig, GeneralDistiller, TrainingConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
sys.path.append('..')
from config import root_path
from generative import config_distil
from generative.bert_model import BertConfig
from generative.optimizer import BERTAdam
from generative.seq2seq import Seq2SeqModel
from generative.tokenizer import Tokenizer, load_chinese_base_vocab
from utils.tools import create_logger, divide_parameters


def read_corpus(data_path):
    df = pd.read_csv(data_path,
                     sep='\t',
                     header=None,
                     names=['src', 'tgt'],
                     quoting=csv.QUOTE_NONE
                    ).dropna()
    sents_src = []
    sents_tgt = []
    for index, row in df.iterrows():
        query = row["src"]
        answer = row["tgt"]
        sents_src.append(query)
        sents_tgt.append(answer)
    return sents_src, sents_tgt


# 自定义dataset
class SelfDataset(Dataset):
    """
    针对数据集，定义一个相关的取数据的方式
    """
    def __init__(self, path, max_length):
        # 一般init函数是加载所有数据
        super(SelfDataset, self).__init__()
        # 读原始数据
        self.sents_src, self.sents_tgt = read_corpus(path)
        self.word2idx = load_chinese_base_vocab()
        self.idx2word = {k: v for v, k in self.word2idx.items()}
        self.tokenizer = Tokenizer(self.word2idx)

        self.max_length = max_length

    def __getitem__(self, i):
        # 得到单个数据

        src = self.sents_src[i] if len(
            self.sents_src[i]
        ) < self.max_length else self.sents_src[i][:self.max_length]
        tgt = self.sents_tgt[i] if len(
            self.sents_tgt[i]
        ) < self.max_length else self.sents_tgt[i][:self.max_length]

        token_ids, token_type_ids = self.tokenizer.encode(src, tgt)
        output = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
        }
        return output

    def __len__(self):

        return len(self.sents_src)


def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """
    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        注意 token type id 右侧pad是添加1而不是0，1表示属于句子B
        """
        pad_indice = [
            item + [pad_idx] * max(0, max_length - len(item))
            for item in indice
        ]
        return torch.tensor(pad_indice).to(args.device)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return token_ids_padded, token_type_ids_padded, target_ids_padded


def main():

 


def load_model(model, pretrain_model_path):

    checkpoint = torch.load(pretrain_model_path, map_location=torch.device('cpu'))
    # 模型刚开始训练的时候, 需要载入预训练的BERT

    checkpoint = {
        k[5:]: v
        for k, v in checkpoint.items() if k[:4] == "bert" and "pooler" not in k
    }
    model.load_state_dict(checkpoint, strict=False)
    torch.cuda.empty_cache()
    logger.info("{} loaded!".format(pretrain_model_path))


if __name__ == "__main__":
    main()
