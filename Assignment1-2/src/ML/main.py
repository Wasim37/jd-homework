'''
@Author: xiaoyao jiang
LastEditors: xiaoyao jiang
@Date: 2020-07-01 15:52:07
LastEditTime: 2020-08-13 23:18:31
FilePath: /bookClassification/src/ML/main.py
@Desciption: Machine Learning model main function
'''
import argparse

from __init__ import *
from src.utils import config
from src.utils.tools import create_logger
from src.ML.models import Models

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--search_method',
                    default='bayesian',
                    type=str,
                    required=False,
                    help='grid / bayesian optimzation')
parser.add_argument('--unbalance',
                    default=True,
                    type=bool,
                    required=False,
                    help='wether use imbalance tech')
parser.add_argument('--imbalance_method',
                    default='other',
                    type=str,
                    required=False,
                    help='under_sampling, over_sampling, ensemble, other')
parser.add_argument('--model_name',
                    default='lgb_under_sampling',
                    type=str,
                    required=False,
                    help='model name')
args = parser.parse_args()

logger = create_logger(config.root_path + '/logs/main.log')

if __name__ == '__main__':
    m = Models(config.root_path + '/model/ml_model/' + args.model_name)
    m.unbalance_helper(imbalance_method=args.imbalance_method,
                       search_method=args.search_method)