'''
@Author: lpx
@Date: 2020-07-17 17:24:00
@LastEditTime: 2020-07-18 16:49:59
@LastEditors: Please set LastEditors
@Description: Testing vocab.
@FilePath: /Assignment1/model/test_vocab.py
'''
import sys
import os
import pathlib

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))
from dataset import PairDataset
import config

dataset = PairDataset(config.data_path,
                        max_src_len=config.max_src_len,
                        max_tgt_len=config.max_tgt_len,
                        truncate_src=config.truncate_src,
                        truncate_tgt=config.truncate_tgt)

vocab = dataset.build_vocab(embed_file=config.embed_file)
test_token = '的'
test_token_idx = vocab['的']
print('idx: ', test_token_idx)
print('token: ', vocab[test_token_idx])
print('vocab size: ', len(vocab))
