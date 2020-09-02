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

    httpClient = None
    myurl = '/api/trans/vip/translate'

    fromLang = source  # The original language
    toLang = target  # The target language
    salt = random.randint(32768, 65536)
    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = '/api/trans/vip/translate' + '?appid=' + appid + '&q=' + urllib.parse.quote(
        q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
        salt) + '&sign=' + sign

    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)

        # response is HTTPResponse object
        response = httpClient.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)

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
    en = translate(q, "zh", 'en')['trans_result'][0]['dst']
    time.sleep(1.5)
    target = translate(en, "en", 'zh')['trans_result'][0]['dst']
    time.sleep(1.5)
    return target

def translate_continue(sample_path, translate_path):
    """translate  original file to new file

    Args:
        sample_path (str): original file path
        translate_path (str): target file path
    Returns:
        (str): result of back translation
    """
    # if file is exist open it ,get length,otherwise build it 
    if os.path.exists(translate_path):
        with open(translate_path, 'r+', encoding='utf8') as file:
            exit_len = len(list(file))
    else:
        with open(translate_path, 'w', encoding='utf8') as file:
            exit_len = 0

    translated = []
    count = 0
    with open(sample_path, 'r', encoding='utf8') as file:
        for line in file:
            count += 1
            print(count)
            if count <= exit_len or count == 21585:
                continue
            # source back_translate  and ref back_translate
            source, ref = tuple(line.strip().split('<sep>'))
            source = back_translate(source.strip())
            ref = back_translate(ref.strip())
            source = ' '.join(list(jieba.cut(source)))
            ref = ' '.join(list(jieba.cut(ref)))
            translated.append(source + ' <sep> ' + ref)
            #  storage back_translate result
            if count % 10 == 0:
                print(count)
                write_samples(translated, translate_path, 'a')
                translated = []
            if count == 12 or count == 25:
                write_samples(translated, translate_path, 'a')
                break




if __name__ == '__main__':
    sample_path = 'output/train.txt'
    translate_path = 'output/translated.txt'
    translate_continue(sample_path, translate_path)
