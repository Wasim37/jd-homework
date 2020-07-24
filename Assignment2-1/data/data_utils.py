#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: lpx
@Date: 2020-07-13 11:00:51
@LastEditTime: 2020-07-16 15:41:42
@LastEditors: Please set LastEditors
@Description: Helper functions or classes used in data processing.
@FilePath: /JD_project_2/baseline/data/data_utils.py
'''


def read_samples(filename):
    samples = []
    with open(filename, 'r', encoding='utf8') as file:
        for line in file:
            samples.append(line.strip())
    return samples


def write_samples(samples, file_path, opt='w'):
    with open(file_path, opt, encoding='utf8') as file:
        for line in samples:
            file.write(line)
            file.write('\n')


def partition(samples):
    """Partition a whole sample set into training set, dev set and test set.

    Args:
        samples (Iterable): The iterable that holds the whole sample set.
    """
    train, dev, test = [], [], []
    count = 0
    for sample in samples:
        count += 1
        if count % 1000 == 0:
            print(count)
        if count <= 1000:
            test.append(sample)
        elif count <= 6000:
            dev.append(sample)
        else:
            train.append(sample)
    print('train: ', len(train))

    write_samples(train, '..files/train.txt')
    write_samples(dev, '..files/dev.txt')
    write_samples(test, '..files/test.txt')


def isChinese(word):
    """Distinguish Chinese words from non-Chinese ones.

    Args:
        word (str): The word to be distinguished.

    Returns:
        bool: Whether the word is a Chinese word.
    """
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False
