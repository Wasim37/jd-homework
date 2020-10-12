#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Peixin Lin
Date: 2020-09-11 11:44:54
LastEditTime: 2020-09-11 15:26:35
FilePath: /Assignment3-2/ranking/train_LM.py
Desciption: Train tfidf, w2v, fasttext models.
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''

import logging
import sys
import os
from collections import defaultdict

import jieba
from gensim import corpora, models

sys.path.append('..')
from config import root_path

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class Trainer(object):
    def __init__(self):
        self.data = self.data_reader(os.path.join(root_path, 'data/ranking/train.tsv')) + \
            self.data_reader(os.path.join(root_path, 'data/ranking/dev.tsv')) + \
            self.data_reader(os.path.join(root_path, 'data/ranking/test.tsv'))
        self.stopwords = open(os.path.join(root_path, 'data/stopwords.txt')).readlines()
        self.preprocessor()
        self.train()
        self.saver()

    def data_reader(self, path):
        samples = []
        with open(path, 'r') as f:
            for line in f:
                try:
                    q1, q2, label = line.split('\t')
                except Exception:
                    print('exception: ', line)
                samples.append(q1)
                samples.append(q2)
        return samples

    def preprocessor(self):
        '''
        @description: 分词， 并生成计算tfidf 所需要的数据
        @param {type}
        @return:
        '''
        logging.info(" loading data.... ")
        self.data = [[
            word for word in jieba.cut(sentence) if word not in self.stopwords
        ] for sentence in self.data]
        self.freq = defaultdict(int)
        for sentence in self.data:
            for word in sentence:
                self.freq[word] += 1
        self.data = [[word for word in sentence if self.freq[word] > 1]
                     for sentence in self.data]
        logging.info(' building dictionary....')
        self.dictionary = corpora.Dictionary(self.data)
        self.dictionary.save(os.path.join(root_path, 'model/ranking/ranking.dict'))
        self.corpus = [self.dictionary.doc2bow(text) for text in self.data]
        corpora.MmCorpus.serialize(os.path.join(root_path, 'model/ranking/ranking.mm'),
                                   self.corpus)

    def train(self):
        

    def saver(self):
        logging.info(' save tfidf model ...')
        self.tfidf.save(os.path.join(root_path, 'model/ranking/tfidf'))
        logging.info(' save word2vec model ...')
        self.w2v.save(os.path.join(root_path, 'model/ranking/w2v'))
        logging.info(' save fasttext model ...')
        self.fast.save(os.path.join(root_path, 'model/ranking/fast'))


if __name__ == "__main__":
    Trainer()
