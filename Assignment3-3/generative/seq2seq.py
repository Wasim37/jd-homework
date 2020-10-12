#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Peixin Lin
Date: 2020-09-29 17:05:17
LastEditTime: 2020-09-30 10:25:06
FilePath: /Assignment3-3/generative/seq2seq.py
Desciption: Using the BERT model for seq2seq task.
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''
import sys

import torch
import torch.nn as nn

from config import max_length

from .bert_model import BertConfig, BertLMPredictionHead, BertModel
from .tokenizer import Tokenizer, load_chinese_base_vocab

sys.path.append('..')


class Seq2SeqModel(nn.Module):
    """
    """
    def __init__(self, config: BertConfig):
        super(Seq2SeqModel, self).__init__()
        self.bert = BertModel(config)
        self.hidden_dim = config.hidden_size
        self.vocab_size = config.vocab_size
        self.decoder = BertLMPredictionHead(
            config, self.bert.embeddings.word_embeddings.weight)
        # 加载字典和分词器
        self.word2ix = load_chinese_base_vocab()
        self.tokenizer = Tokenizer(self.word2ix)

    def compute_loss(self, predictions, labels, target_mask):
        """
        target_mask : 句子a部分和pad部分全为0， 而句子b部分为1
        """
       
        return 

    def forward(self,
                input_tensor,
                token_type_id,
                labels=None,
                position_enc=None,
                is_cuda=True):
        
        if labels is not None:
            # 计算loss
            # 需要构建特殊的输出mask 才能计算正确的loss
            # 预测的值不用取最后sep符号的结果 因此是到-1
            
            return predictions, loss
        else:
            return predictions

    def generate(self, text, out_max_length=50, beam_size=1, is_cuda=False):
        # 对 一个 句子生成相应的结果
        # 通过输出最大长度得到输入的最大长度，这里问题不大，如果超过最大长度会进行截断

        # 解码 得到相应输出
        return self.tokenizer.decode(out_puts_ids)

    def beam_search(self,
                    token_ids,
                    token_type_ids,
                    word2ix,
                    beam_size=1,
                    is_cuda=False,
                    alpha=0.5):
        """
        beam-search操作
        """

        return output_ids[output_scores.argmax().item()]
