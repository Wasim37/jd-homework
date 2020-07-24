#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: lpx, jby
@Date: 2020-07-13 11:07:48
@LastEditTime: 2020-07-18 14:25:15
@LastEditors: Please set LastEditors
@Description: Helper functions or classes used for the model.
@FilePath: /JD_project_2/baseline/model/utils.py
'''

import time
import heapq

import numpy as np


def timer(module):
    """Decorator function for a timer.

    Args:
        module (str): Description of the function being timed.
    """
    def wrapper(func):
        """Wrapper of the timer function.

        Args:
            func (function): The function to be timed.
        """
        def cal_time(*args, **kwargs):
            """The timer function.

            Returns:
                res (any): The returned value of the function being timed.
            """
            t1 = time.time()
            res = func(*args, **kwargs)
            t2 = time.time()
            cost_time = t2 - t1
            print(f'{cost_time} secs used for ', module)
            return res
        return cal_time
    return wrapper


def simple_tokenizer(text):
    return text.split()


def count_words(counter, text):
    '''Count the number of occurrences of each word in a set of text'''
    for sentence in text:
        for word in sentence:
            counter[word] += 1


def sort_batch_by_len(data_batch):
    """

    Args:
        data_batch (Tensor): Batch before sorted.

    Returns:
        Tensor: Batch after sorted.
    """
    res = {'x': [],
           'y': [],
           'x_len': [],
           'y_len': [],
           'OOV': [],
           'len_OOV': []}
    for i in range(len(data_batch)):
        res['x'].append(data_batch[i]['x'])
        res['y'].append(data_batch[i]['y'])
        res['x_len'].append(len(data_batch[i]['x']))
        res['y_len'].append(len(data_batch[i]['y']))
        res['OOV'].append(data_batch[i]['OOV'])
        res['len_OOV'].append(data_batch[i]['len_OOV'])

    # Sort indices of data in batch by lengths.
    sorted_indices = np.array(res['x_len']).argsort()[::-1].tolist()

    data_batch = {
        name: [_tensor[i] for i in sorted_indices]
        for name, _tensor in res.items()
    }
    return data_batch


def outputids2words(id_list, source_oovs, vocab):
    """
        Maps output ids to words, including mapping in-source OOVs from
        their temporary ids to the original OOV string (applicable in
        pointer-generator mode).
        Args:
            id_list: list of ids (integers)
            vocab: Vocabulary object
            source_oovs:
                list of OOV words (strings) in the order corresponding to
                their temporary source OOV ids (that have been assigned in
                pointer-generator mode), or None (in baseline mode)
        Returns:
            words: list of words (strings)
    """

    ###########################################
    #          TODO: module 1 task 4          #
    ###########################################

    words = []

    return ' '.join(words)


def source2ids(source_words, vocab):
    """Map the source words to their ids and return a list of OOVs in the source.
    Args:
        source_words: list of words (strings)
        vocab: Vocabulary object
    Returns:
        ids:
        A list of word ids (integers); OOVs are represented by their temporary
        source OOV number. If the vocabulary size is 50k and the source has 3
        OOVs tokens, then these temporary OOV numbers will be 50000, 50001,
        50002.
    oovs:
        A list of the OOV words in the source (strings), in the order
        corresponding to their temporary source OOV numbers.
    """

    ###########################################
    #          TODO: module 1 task 3          #
    ###########################################

    ids = []
    oovs = []
    unk_id = vocab["<UNK>"]
 
    return ids, oovs


class Beam(object):
    """The contianer for a temperay sequence used in beam search.
    """
    def __init__(self,
                 tokens,
                 log_probs,
                 decoder_states,
                 attention_weights,
                 max_oovs,
                 encoder_input):
        self.tokens = tokens
        self.log_probs = log_probs
        self.decoder_states = decoder_states
        self.attention_weights = attention_weights
        self.max_oovs = max_oovs
        self.encoder_input = encoder_input

    def extend(self,
               token,
               log_prob,
               decoder_states,
               attention_weights,
               max_oovs,
               encoder_input):
        """Extend the curren beam using now token and return a new beam.
        """
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    decoder_states=decoder_states,
                    attention_weights=attention_weights,
                    max_oovs=max_oovs,
                    encoder_input=encoder_input)

    def seq_score(self):
        """
        This function calculates the score of the current sequence.
        """
        score = sum(self.log_probs) / len(self.tokens)
        return score.item()

    def __lt__(self, other):
        return self.seq_score() < other.seq_score()

    def __le__(self, other):
        return self.seq_score() <= other.seq_score()


def add2heap(heap, item, k):
    """Maintain a heap with k nodes and the smallest one as root.

    Args:
        heap (list): The list to heapify.
        item (tuple):
            The tuple as item to store.
            Comparsion will be made according to values in the first position.
            If there is a tie, values in the second position will be compared,
            and so on.
        k (int): The capacity of the heap.
    """
    if len(heap) < k:
        heapq.heappush(heap, item)
    else:
        heapq.heappushpop(heap, item)
