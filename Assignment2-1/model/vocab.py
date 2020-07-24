#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: jby
@Date: 2020-07-13 14:18:13
@LastEditTime: 2020-07-18 14:17:37
@LastEditors: Please set LastEditors
@Description: Define the vocabulary object.
@FilePath: /JD_project_2/baseline/model/vocab.py
'''

from collections import Counter

import numpy as np


class Vocab(object):
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3

    def __init__(self):
        self.word2index = {}
        self.word2count = Counter()
        self.reserved = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        self.index2word = self.reserved[:]
        self.embeddings = None

    def add_words(self, words):
        """Add a new token to the vocab and do mapping between word and index.

        Args:
            words (list): The list of tokens to be added.
        """
        ###########################################
        #          TODO: module 1 task 1          #
        ###########################################

    def load_embeddings(self, file_path: str, dtype=np.float32) -> int:
        """Load embedding word vector.

        Args:
            file_path (str): The file path of word vector to load.
            dtype (numpy dtype, optional): Defaults to np.float32.

        Returns:
            int: Number of embedded tokens.
        """
        num_embeddings = 0
        vocab_size = len(self)
        with open(file_path, 'rb') as f:
            for line in f:
                line = line.split()
                word = line[0].decode('utf-8')
                idx = self.word2index.get(word)
                # Check whether the token is in the vocab.
                if idx is not None:
                    vec = np.array(line[1:], dtype=dtype)
                    if self.embeddings is None:
                        # Get embedding dimension.
                        n_dims = len(vec)
                        # Initialize word vectors.
                        self.embeddings = np.random.normal(
                            np.zeros((vocab_size, n_dims))).astype(dtype)
                        self.embeddings[self.PAD] = np.zeros(n_dims)
                    self.embeddings[idx] = vec
                    num_embeddings += 1
        return num_embeddings

    def __getitem__(self, item):
        if type(item) is int:
            return self.index2word[item]
        return self.word2index.get(item, self.UNK)

    def __len__(self):
        return len(self.index2word)

    def size(self):
        """Returns the total size of the vocabulary"""
        return len(self.index2word)
