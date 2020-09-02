#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: lpx
@Date: 2020-07-13 20:16:37
@LastEditTime: 2020-07-18 17:28:41
@LastEditors: Please set LastEditors
@Description: 
@FilePath: /JD_project_2/data/embed_replace.py
@Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''

from gensim.models import KeyedVectors, TfidfModel
from gensim.corpora import Dictionary
from data_utils import read_samples, isChinese, write_samples
import os
from gensim import matutils
from itertools import islice
import numpy as np


class EmbedReplace():
    def __init__(self, sample_path, wv_path):
        self.samples = read_samples(sample_path)
        self.refs = [sample.split('<sep>')[1].split() for sample in self.samples]
        self.wv = KeyedVectors.load_word2vec_format(
            wv_path,
            binary=False)

        if os.path.exists('saved/tfidf.model'):
            self.tfidf_model = TfidfModel.load('saved/tfidf.model')
            self.dct = Dictionary.load('saved/tfidf.dict')
            self.corpus = [self.dct.doc2bow(doc) for doc in self.refs]
        else:
            self.dct = Dictionary(self.refs)
            self.corpus = [self.dct.doc2bow(doc) for doc in self.refs]
            self.tfidf_model = TfidfModel(self.corpus)
            self.dct.save('saved/tfidf.dict')
            self.tfidf_model.save('saved/tfidf.model')
            self.vocab_size = len(self.dct.token2id)

    def vectorize(self, docs, vocab_size):
        '''
        docs :: iterable of iterable of (int, number)
        '''
        return matutils.corpus2dense(docs, vocab_size)

    def extract_keywords(self, dct, tfidf, threshold=0.2, topk=5):

        """find high TFIDF socore keywords

        Args:
            dct (Dictionary): gensim.corpora Dictionary  a reference Dictionary
            tfidf (list of tfidf):  model[doc]  [(int, number)]
            threshold (float) : high TFIDF socore must be greater than the threshold
            topk(int): num of highest TFIDF socore 
        Returns:
            (list): A list of keywords
        """

        tfidf = sorted(tfidf, key=lambda x: x[1], reverse=True)
        return list(islice(
            [dct[w] for w, score in tfidf if score > threshold], topk
            ))

    def replace(self, token_list, doc):
        """replace token by another token which is similar in wordvector 

        Args:
            token_list (list): reference token list
            doc (list): A reference represented by a word bag model
        Returns:
            (str):  new reference str
        """
        keywords = self.extract_keywords(self.dct, self.tfidf_model[doc])
        num = int(len(token_list) * 0.3)
        new_tokens = token_list.copy()
        while num == int(len(token_list) * 0.3):
            indexes = np.random.choice(len(token_list), num)
            for index in indexes:
                token = token_list[index]
                if isChinese(token) and token not in keywords and token in self.wv:
                    new_tokens[index] = self.wv.most_similar(
                        positive=token, negative=None, topn=1
                        )[0][0]
            num -= 1

        return ' '.join(new_tokens)

    def generate_samples(self, write_path):
        """generate new samples file
        Args:
            write_path (str):  new samples file path

        """
        replaced = []
        count = 0
        for sample, token_list, doc in zip(self.samples, self.refs, self.corpus):
            count += 1
            if count % 100 == 0:
                print(count)
                write_samples(replaced, write_path, 'a')
                replaced = []
            replaced.append(
                sample.split('<sep>')[0] + ' <sep> ' + self.replace(token_list, doc)
                )


sample_path = 'output/train.txt'
wv_path = 'word_vectors/merge_sgns_bigram_char300.txt'
replacer = EmbedReplace(sample_path, wv_path)
replacer.generate_samples('output/replaced.txt')
