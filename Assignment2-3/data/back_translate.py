#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@Author: frank
@Date: 2020-08-08 22:40:43
@LastEditTime: 
@LastEditors: 
@Description: 
@File: back_translate.py
@Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
"""
# !pip3 install jieba==0.36.2
import jieba
import http.client
import hashlib
import urllib
import random
import json
import time
from data_utils import write_samples

import os


def translate(q, source, target):
    """translate q from source language to target language

    Args:
        q (str): sentence
        source(str): The language code
        target(str): The language code
    Returns:
        (str): result of translation
    """
    # Please refer to the official documentation   https://api.fanyi.baidu.com/  
    # There are demo on the website ,  register on the web site ,and get AppID, key, python3 demo.
    appid = ''  # Fill in your AppID
    secretKey = ''  # Fill in your key

    ###########################################
    #          TODO: module 2 task 1          #
    ###########################################

        return result

    except Exception as e:
        print(e)
    finally:
        if httpClient:
            httpClient.close()


def back_translate(q):
    """back_translate

    Args:
        q (str): sentence

    Returns:
        (str): result of back translation
    """
    ###########################################
    #          TODO: module 2 task 2          #
    ###########################################
    return target

def translate_continue(sample_path, translate_path):
    """translate  original file to new file

    Args:
        sample_path (str): original file path
        translate_path (str): target file path
    Returns:
        (str): result of back translation
    """
    ###########################################
    #          TODO: module 2 task 3          #
    ###########################################




if __name__ == '__main__':
    sample_path = 'output/train.txt'
    translate_path = 'output/translated.txt'
    translate_continue(sample_path, translate_path)