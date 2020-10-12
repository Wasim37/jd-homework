#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Peixin Lin
Date: 2020-09-11 11:44:54
LastEditTime: 2020-09-11 15:35:16
FilePath: /Assignment3-2/ranking/similarity.py
Desciption: Definition of manual features.
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''

import logging
import sys
import os

import jieba.posseg as pseg
import numpy as np
from gensim import corpora, models

from config import root_path
from retrieval.hnsw_faiss import wam
from ranking.bm25 import BM25

sys.path.append('..')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class TextSimilarity(object):
    def __init__(self):
        logging.info('load dictionary')
        self.dictionary = corpora.Dictionary.load(os.path.join(root_path,
                                                  'model/ranking/ranking.dict'))
        logging.info('load corpus')
        self.corpus = corpora.MmCorpus(os.path.join(root_path, 'model/ranking/ranking.mm'))
        logging.info('load tfidf')
        self.tfidf = models.TfidfModel.load(os.path.join(root_path, 'model/ranking/tfidf'))
        logging.info('load bm25')
        self.bm25 = BM25(do_train=False)
        logging.info('load word2vec')
        self.w2v_model = models.KeyedVectors.load(os.path.join(root_path, 'model/ranking/w2v'))
        logging.info('load fasttext')
        self.fasttext = models.FastText.load(os.path.join(root_path, 'model/ranking/fast'))

    # get LCS(longest common subsquence),DP
    def lcs(self, str_a, str_b):
        """Longest common substring

        Returns:
            ratio: The length of LCS divided by the length of
                the shorter one among two input strings.
        """

        return ratio

    def editDistance(self, str1, str2):
        """Edit distance

        Returns:
            ratio: Minimum edit distance divided by the length sum
                of two input strings.
        """

        return ratio

    @classmethod
    def tokenize(self, str_a):
        '''
        接受一个字符串作为参数，返回分词后的结果字符串(空格隔开)和集合类型
        '''
        wordsa = pseg.cut(str_a)
        cuta = ""
        seta = set()
        for key in wordsa:
            cuta += key.word + " "
            seta.add(key.word)

        return [cuta, seta]

    def JaccardSim(self, str_a, str_b):
        '''
        Jaccard相似性系数
        计算sa和sb的相似度 len（sa & sb）/ len（sa | sb）
        '''

        return jaccard_sim

    @staticmethod
    def cos_sim(a, b):

        return cos_sim

    @staticmethod
    def eucl_sim(a, b):

        return eucl_sim

    @staticmethod
    def pearson_sim(a, b):

        return pearson_sim

    def tokenSimilarity(self, str_a, str_b, method='w2v', sim='cos'):
        '''
        基于分词求相似度，默认使用cos_sim 余弦相似度,默认使用前20个最频繁词项进行计算
        method: w2v, tfidf, fasttext
        sim: cos, pearson, eucl
        '''

        return result

    def generate_all(self, str1, str2):
        return {
            'lcs':
            self.lcs(str1, str2),
            'edit_dist':
            self.editDistance(str1, str2),
            'jaccard':
            self.JaccardSim(str1, str2),
            'bm25':
            self.bm25.bm_25(str1, str2),
            'w2v_cos':
            self.tokenSimilarity(str1, str2, method='w2v', sim='cos'),
            'w2v_eucl':
            self.tokenSimilarity(str1, str2, method='w2v', sim='eucl'),
            'w2v_pearson':
            self.tokenSimilarity(str1, str2, method='w2v', sim='pearson'),
            'w2v_wmd':
            self.tokenSimilarity(str1, str2, method='w2v', sim='wmd'),
            'fast_cos':
            self.tokenSimilarity(str1, str2, method='fasttest', sim='cos'),
            'fast_eucl':
            self.tokenSimilarity(str1, str2, method='fasttest', sim='eucl'),
            'fast_pearson':
            self.tokenSimilarity(str1,
                                 str2,
                                 method='fasttest',
                                 sim='pearson'),
            'fast_wmd':
            self.tokenSimilarity(str1, str2, method='fasttest', sim='wmd'),
            'tfidf_cos':
            self.tokenSimilarity(str1, str2, method='tfidf', sim='cos'),
            'tfidf_eucl':
            self.tokenSimilarity(str1, str2, method='tfidf', sim='eucl'),
            'tfidf_pearson':
            self.tokenSimilarity(str1, str2, method='tfidf', sim='pearson')
        }
