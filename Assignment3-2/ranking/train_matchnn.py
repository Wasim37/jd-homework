#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Peixin Lin
Date: 2020-09-11 11:44:54
LastEditTime: 2020-09-11 15:28:20
FilePath: /Assignment3-2/ranking/train_matchnn.py
Desciption: Train a matching network.
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''

import os
import torch
from torch.utils.data import DataLoader
from data import DataPrecessForSentence
from matchnn_utils import train, validate
from transformers import BertTokenizer
from matchnn import BertModelTrain
from transformers.optimization import AdamW
import sys
sys.path.append('..')
from config import is_cuda, root_path, max_sequence_length

seed = 9
torch.manual_seed(seed)
if is_cuda:
    torch.cuda.manual_seed_all(seed)


def main(train_file,
         dev_file,
         target_dir,
         epochs=10,
         batch_size=32,
         lr=2e-05,
         patience=3,
         max_grad_norm=10.0,
         checkpoint=None):


if __name__ == "__main__":
    main(os.path.join(root_path, 'data/ranking/train.tsv'),
         os.path.join(root_path, 'data/ranking/dev.tsv'),
         os.path.join(root_path, "model/ranking/"))

