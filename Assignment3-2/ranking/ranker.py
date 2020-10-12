#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Peixin Lin
Date: 2020-09-11 11:44:54
LastEditTime: 2020-09-11 15:39:17
FilePath: /Assignment3-2/ranking/ranker.py
Desciption: Generating features and train a LightGBM ranker.
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''

import sys
import os
import csv
import logging

import lightgbm as lgb
import pandas as pd
import joblib
from tqdm import tqdm

sys.path.append('..')
from config import root_path
from ranking.matchnn import MatchingNN
from ranking.similarity import TextSimilarity
from retrieval.hnsw_faiss import wam

from sklearn.model_selection import train_test_split
import numpy as np

tqdm.pandas()

# Parameters for lightGBM
params = {


}


class RANK(object):
    def __init__(self, do_train=True,  model_path=os.path.join(root_path, 'model/ranking/lightgbm')):
        self.ts = TextSimilarity()
        self.matchingNN = MatchingNN()
        if do_train:
            logging.info('Training mode')

        else:
            logging.info('Predicting mode')


    def generate_feature(self, data):
        '''
        @description: 生成模型训练所需要的特征
        @param {type}
        data Dataframe
        @return: Dataframe
        '''

        return data

    def trainer(self):
        logging.info('Training lightgbm model.')


    def save(self, model_path):
        logging.info('Saving lightgbm model.')
        joblib.dump(self.gbm, model_path)

    def predict(self, data: pd.DataFrame):
        """Doing prediction.

        Args:
            data (pd.DataFrame): the output of self.generate_feature

        Returns:
            result[list]: The scores of all query-candidate pairs.
        """
        return result


if __name__ == "__main__":
    rank = RANK(do_train=True)

