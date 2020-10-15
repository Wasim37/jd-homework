#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Wasim37
Date: 2020-10-05 15:53:39
'''

import logging
import sys
import os
import random
import pandas as pd
import csv

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.split(curPath)[0])
from config import root_path, atec_nlp_sim_train, atec_nlp_sim_train_add, task3_train


def generate_data(to_file=os.path.join(root_path, 'data/ranking')):
    '''
    @description: 整合并拆分数据集
    @param {type}
    filedir 文件目录
    @return:
    '''
    logging.info("loading data.... ")
    df_list = []
    with open(atec_nlp_sim_train, 'r', encoding='UTF-8') as f:
        for lines in f:
            line_list = []
            line = lines.strip().split('\t')
            if len(line) != 4:
                continue
            else:
                line_list.append(line[1])
                line_list.append(line[2])
                line_list.append(line[3])
                df_list.append(line_list)

    with open(atec_nlp_sim_train_add, 'r', encoding='UTF-8') as f:
        for lines in f:
            line_list = []
            line = lines.strip().split('\t')
            if len(line) != 4:
                continue
            else:
                line_list.append(line[1])
                line_list.append(line[2])
                line_list.append(line[3])
                df_list.append(line_list)

    with open(task3_train, 'r', encoding='UTF-8') as f:
        for lines in f:
            line_list = []
            line = lines.strip().split('\t')
            if len(line) != 3:
                continue
            else:
                line_list.append(line[0])
                line_list.append(line[1])
                line_list.append(line[2])
                df_list.append(line_list)

    random.shuffle(df_list)
    num = len(df_list)
    train_offset = int(num * 0.8)
    dev_offset = int(num * 0.1)
    train_list = df_list[:train_offset]
    dev_list = df_list[train_offset: train_offset + dev_offset]
    test_list = df_list[train_offset + dev_offset:]

    logging.info("save data.... ：train.size:{}, test.size:{}, dev.size:{}...".format(len(train_list), len(test_list), len(dev_list)))
    pd.DataFrame(train_list).to_csv(to_file + '/train.tsv', index=False, header=False, encoding="utf-8", quoting=csv.QUOTE_NONE, sep='\t')
    pd.DataFrame(test_list).to_csv(to_file + '/test.tsv', index=False, header=False, encoding="utf-8", quoting=csv.QUOTE_NONE, sep='\t')
    pd.DataFrame(dev_list).to_csv(to_file + '/dev.tsv', index=False, header=False, encoding="utf-8", quoting=csv.QUOTE_NONE, sep='\t')


if __name__ == "__main__":
    dev = generate_data()
