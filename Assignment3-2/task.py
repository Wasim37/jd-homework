#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Peixin Lin
Date: 2020-09-11 14:14:22
LastEditTime: 2020-09-11 15:40:14
FilePath: /Assignment3-2/task.py
Desciption: Combine intention module, retrieval module
    and ranking module for task-oriented dialogue.
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''
import os

from intention.business import Intention
from retrieval.hnsw_faiss import HNSW
from ranking.ranker import RANK
import config
import pandas as pd


def retrieve(k):

    res.to_csv('result/retrieved.csv', index=False)


def rank():
    
    ranked.to_csv('result/ranked.csv', index=False)


if __name__ == "__main__":
    retrieve(5)
    rank()
