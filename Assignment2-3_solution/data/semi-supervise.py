#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@Author: frank
@Date: 2020-08-07 16:43:30
@LastEditTime: 
@LastEditors: 
@Description: 
@File: semi-supervised.py
@Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
"""
import pathlib
import sys

abs_path = pathlib.Path(__file__).parent
sys.path.append('../model')

from predict import Predict
from data_utils import write_samples





def semi_supervised(samples_path, write_path, beam_search):
    """use reference to predict source

    Args:
        samples_path (str): The path of reference
        write_path (str): The path of new samples

    """
    pred = Predict()
    print('vocab_size: ', len(pred.vocab))
    # Randomly pick a sample in test set to predict.
    count = 0
    semi = []

    with open(samples_path, 'r') as f:
        for picked in f:
            count += 1
            source, ref = picked.strip().split('<sep>')
            prediction = pred.predict(ref.split(), beam_search=beam_search)
            semi.append(prediction + ' <sep> ' + ref)

            if count % 100 == 0:
                print(count)
                write_samples(semi, write_path, 'a')
                semi = []

if __name__ == '__main__':
    samples_path = 'output/train.txt'
    write_path_greedy = 'output/semi_greedy.txt'
    write_path_beam = 'output/semi_beam.txt'
    beam_search = False
    if beam_search:
        write_path = write_path_beam
    else:
        write_path = write_path_greedy
    semi_supervised(samples_path, write_path, beam_search)
