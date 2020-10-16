#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Please set LastEditors
Date: 2020-09-11 14:14:22
LastEditTime: 2020-10-16 12:39:05
FilePath: /Assignment3-2_solution/task.py
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

    it = Intention(config.train_path,
                   config.ware_path,
                   model_path=config.ft_path,
                   kw_path=config.keyword_path)

    hnsw = HNSW(config.w2v_path,
                config.ef_construction,
                config.M,
                config.hnsw_path,
                config.train_path)

    dev_set = pd.read_csv(os.path.join(config.root_path, 'data/dev.csv')).dropna()
    test_set = pd.read_csv(os.path.join(config.root_path, 'data/test.csv')).dropna()
    data_set = dev_set.append(test_set)

    res = pd.DataFrame()
    for query in data_set['custom']:
        query = query.strip()
        # 意图识别
        intention = it.predict(query)
        if len(query) > 1 and intention == '__label__1':
            res = res.append(
                pd.DataFrame({
                    'query': [query] * k,
                    # 每个query对应k个召回结果
                    'retrieved': hnsw.search(query, k)['custom']
                }))

    res.to_csv('result/retrieved.csv', index=False)


def rank():
    """对result/retrieved.csv中每个query-retrieved pair进行评分
       并记录在新的column中
    """
    retrieved = pd.read_csv(
        os.path.join(config.root_path, 'result/retrieved.csv'))
    ranker = RANK(do_train=False)
    ranked = pd.DataFrame()
    ranked['question1'] = retrieved['query']
    ranked['question2'] = retrieved['retrieved']
    feature_data = ranker.generate_feature(ranked)
    rank_scores = ranker.predict(feature_data)
    ranked['rank_score'] = rank_scores
    ranked.to_csv('result/ranked.csv', index=False)


if __name__ == "__main__":
    retrieve(4)
    rank()
