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

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.split(curPath)[0])
from config import root_path, atec_nlp_sim_train, atec_nlp_sim_train_add, task3_train


def generate_data(to_file):
    '''
    @description: 整合并拆分数据集
    @param {type}
    filedir 文件目录
    @return:
    '''
    logging.info("开始读取数据...")
    df_list = []
    with open(atec_nlp_sim_train, 'r', encoding='utf8') as f:
        for lines in f:
            line = lines.strip().split('\t')
            if len(line) != 4:
                continue
            else:
                df_list.append(line[1] + '\t' + line[2] + '\t' + line[3])

    with open(atec_nlp_sim_train_add, 'r', encoding='utf8') as f:
        for lines in f:
            line = lines.strip().split('\t')
            if len(line) != 4:
                continue
            else:
                df_list.append(line[1] + '\t' + line[2] + '\t' + line[3])

    with open(task3_train, 'r', encoding='utf8') as f:
        for lines in f:
            line = lines.strip().split('\t')
            if len(line) != 3:
                continue
            else:
                df_list.append(line[0] + '\t' + line[1] + '\t' + line[2])

    logging.info("开始拆分数据...")
    random.shuffle(df_list)
    num = len(df_list)
    train_offset = int(num * 0.8)
    dev_offset = int(num * 0.1)
    train_list = df_list[:train_offset]
    dev_list = df_list[train_offset: train_offset + dev_offset]
    test_list = df_list[train_offset + dev_offset:]

    logging.info("开始存储数据：train.size:{}, test.size:{}, dev.size:{}...".format(len(train_list), len(test_list), len(dev_list)))
    pd.DataFrame(train_list).to_csv(to_file + '/train.csv', index=False, header=False)
    pd.DataFrame(test_list).to_csv(to_file + '/test.csv', index=False, header=False)
    pd.DataFrame(dev_list).to_csv(to_file + '/dev.csv', index=False, header=False)


if __name__ == "__main__":
    dev = generate_data(to_file=os.path.join(root_path, 'data/ranking'))
