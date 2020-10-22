#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Please set LastEditors
Date: 2020-09-29 17:05:16
LastEditTime: 2020-10-22 15:43:24
FilePath: /Assignment3-3/generative/data.py
Desciption: Data processing for the generative module.
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''

import logging
import sys
import os
import json

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.split(curPath)[0])
from config import root_path, LCCC_base_train_json, LCCC_base_dev_json, LCCC_base_test_json


def generate_data(to_file=os.path.join(root_path, 'data/generative/')):
    
    logging.info("generating train data.... ")
    with open(to_file + "train.tsv", 'w', encoding='UTF-8') as train:
        count = 0
        with open(LCCC_base_train_json, 'r', encoding='UTF-8') as f:
            json_file = json.load(f)
            for item in json_file:
                if count > 100000:
                    break
                if len(item) == 2:
                    line = '\t'.join(item)
                    train.write(line)
                    train.write('\n')
                    count += 1
    logging.info("train data size:", count)

    logging.info("generating dev data.... ")
    with open(to_file + "dev.tsv", 'w', encoding='UTF-8') as dev:
        count = 0
        with open(LCCC_base_dev_json, 'r', encoding='UTF-8') as f:
            json_file = json.load(f)
            for item in json_file:
                if len(item) == 2:
                    line = '\t'.join(item)
                    dev.write(line)
                    dev.write('\n')
                    count += 1
    logging.info("dev data size:", count)

    logging.info("generating test data.... ")
    with open(to_file + "test.tsv", 'w', encoding='UTF-8') as test:
        count = 0
        with open(LCCC_base_test_json, 'r', encoding='UTF-8') as f:
            json_file = json.load(f)
            for item in json_file:
                if len(item) == 2:
                    line = '\t'.join(item)
                    test.write(line)
                    test.write('\n')
                    count += 1
    logging.info("test data size:", count)


if __name__ == "__main__":
    dev = generate_data()
