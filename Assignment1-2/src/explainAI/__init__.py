'''
@Author: xiaoyao jiang
@LastEditors: xiaoyao jiang
@Date: 2020-07-13 10:47:19
@LastEditTime: 2020-07-13 10:47:19
@FilePath: /bookClassification/src/explainAI/__init__.py
@Desciption:
'''
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])