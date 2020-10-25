#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Please set LastEditors
Date: 2020-08-21 16:20:49
LastEditTime: 2020-10-23 10:49:50
FilePath: /Assignment3-1_solution/config.py
Desciption: 配置文件。
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''

import torch
import os
root_path = os.path.abspath(os.path.dirname(__file__))

train_raw = os.path.join(root_path, 'data/chat.txt')
dev_raw = os.path.join(root_path, 'data/开发集.txt')
test_raw = os.path.join(root_path, 'data/测试集.txt')
ware_path = os.path.join(root_path, 'data/ware.txt')

atec_nlp_sim_train = os.path.join(root_path, 'data/ranking_datasets/atec_nlp_sim_train.csv')
atec_nlp_sim_train_add = os.path.join(root_path, 'data/ranking_datasets/atec_nlp_sim_train_add.csv')
task3_train = os.path.join(root_path, 'data/ranking_datasets/task3_train.txt')

LCCC_base_train_json = os.path.join(root_path, 'data/generative/LCCC-base_train.json')
LCCC_base_dev_json = os.path.join(root_path, 'data/generative/LCCC-base_valid.json')
LCCC_base_test_json = os.path.join(root_path, 'data/generative/LCCC-base_test.json')

max_sequence_length = 512


sep = '[SEP]'

''' Data '''
# main
train_path = os.path.join(root_path, 'data/train_no_blank.csv')
dev_path = os.path.join(root_path, 'data/dev.csv')
test_path = os.path.join(root_path, 'data/test.csv')
# intention
business_train = os.path.join(root_path, 'data/intention/business.train')
business_test = os.path.join(root_path, 'data/intention/business.test')
keyword_path = os.path.join(root_path, 'data/intention/key_word.txt')


''' Intention '''
# fasttext
ft_path = os.path.join(root_path, "model/intention/fastext")

''' Retrival '''
# Embedding
w2v_path = os.path.join(root_path, "model/retrieval/word2vec")

# HNSW parameters
ef_construction = 3000  # ef_construction defines a construction time/accuracy trade-off
M = 64  # M defines tha maximum number of outgoing connections in the graph
hnsw_path = os.path.join(root_path, 'model/retrieval/hnsw_index')
hnsw_hnswlib_path = os.path.join(root_path, 'model/retrieval/hnsw.bin')

# 通用配置
is_cuda = True
if is_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# 以下作业3新增的配置

# generate config
bert_chinese_model_path = os.path.join(root_path, "lib/bert/pytorch_model.bin")

base_chinese_bert_vocab = os.path.join(root_path, "lib/bert/vocab.txt")
log_path = os.path.join(root_path, "log/seq2seq1.log")
max_sequence_length = 200
max_length = 128
lr = 0.0001
batch_size = 32
gradient_accumulation = 1
max_grad_norm = 1.0
# 数据路径
# main
train_path = os.path.join(root_path, 'data/train.csv')
dev_path = os.path.join(root_path, 'data/dev.csv')
test_path = os.path.join(root_path, 'data/test.csv')
# intention
business_train = os.path.join(root_path, 'data/intention/business.train')
business_test = os.path.join(root_path, 'data/intention/business.test')
keyword_path = os.path.join(root_path, 'data/intention/key_word.txt')

# reasonable config
rea_pretrained_model_name = os.path.join(root_path, 'lib/bert/pytorch_model.bin')
rea_pretrained_model_config = os.path.join(root_path, 'lib/bert/config.json')
rea_chinese_bert_vocab = os.path.join(root_path, "lib/bert/vocab.txt")
rea_train_path = os.path.join(root_path, 'data/bert_classifier/train.tsv')
rea_dev_path = os.path.join(root_path, 'data/bert_classifier/train.tsv')
rea_test_path = os.path.join(root_path, 'data/bert_classifier/train.tsv')
rea_save_dir = os.path.join(root_path, 'model/reasonable/')

# evaluate config
# distill_generate_model = root_path + '/model/bert.model.epoch'
distill_generate_model = os.path.join(root_path, 'model/bert.distilled')
matching_model = os.path.join(root_path, 'model/matching.12.bin')
rea_trained_model_name = os.path.join(root_path, 'model/reasonable/')
intent_detect_model = os.path.join(root_path, 'model/business.model')