'''
@Author: xiaoyao jiang
@Date: 2020-04-08 19:39:30
LastEditTime: 2020-08-13 23:20:30
LastEditors: xiaoyao jiang
@Description: There are two options. One is using pretrained embedding as feature to compare common ML models.
              The other is using feature engineering + param search tech + imbanlance to train a liaghtgbm model.
FilePath: /bookClassification/src/ML/models.py
'''
import os

import lightgbm as lgb
import numpy as np
import torchvision
import json
import pandas as pd
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from transformers import BertModel, BertTokenizer

from __init__ import *
from src.data.mlData import MLData
from src.utils import config
from src.utils.config import root_path
from src.utils.tools import (Grid_Train_model, bayes_parameter_opt_lgb,
                             query_cut, create_logger, formate_data, get_score)
from src.utils.feature import (get_embedding_feature, get_img_embedding,
                               get_lda_features, get_pretrain_embedding,
                               get_autoencoder_feature, get_basic_feature)

logger = create_logger(config.log_dir + 'model.log')


class Models(object):
    def __init__(self,
                 model_path=None,
                 feature_engineer=False,
                 train_mode=True):
        '''
        @description: initlize Class, EX: model
        @param {type} :
        feature_engineer: whether using feature engineering, if `False`, then compare common ML models
        res_model: res network model
        resnext_model: resnext network model
        wide_model: wide res network model
        bert: bert model
        ml_data: new mldata class
        @return: No return
        '''
        # 加载图像处理模型， resnet, resnext, wide resnet， 如果支持cuda, 则将模型加载到cuda中
        self.res_model = torchvision.models.resnet152(
            pretrained=True)  # res model for modal feature [1* 1000]
        self.res_model = self.res_model.to(config.device)
        self.resnext_model = torchvision.models.resnext101_32x8d(
            pretrained=True)
        self.resnext_model = self.resnext_model.to(config.device)
        self.wide_model = torchvision.models.wide_resnet101_2(pretrained=True)
        self.wide_model = self.wide_model.to(config.device)
        # 加载 bert 模型， 如果支持cuda, 则将模型加载到cuda中
        self.bert_tonkenizer = BertTokenizer.from_pretrained(config.root_path +
                                                             '/model/bert')
        self.bert = BertModel.from_pretrained(config.root_path + '/model/bert')
        self.bert = self.bert.to(config.device)

        # 初始化 MLdataset 类， debug_mode为true 则使用部分数据， train_mode表示是否训练
        self.ml_data = MLData(debug_mode=True, train_mode=train_mode)
        # 如果不训练， 则加载训练好的模型，进行预测
        if train_mode:
            self.model = lgb.LGBMClassifier(objective='multiclass',
                                            n_jobs=10,
                                            num_class=33,
                                            num_leaves=30,
                                            reg_alpha=10,
                                            reg_lambda=200,
                                            max_depth=3,
                                            learning_rate=0.05,
                                            n_estimators=2000,
                                            bagging_freq=1,
                                            bagging_fraction=0.9,
                                            feature_fraction=0.8,
                                            seed=1440)

        else:
            self.load(model_path)
            labelNameToIndex = json.load(
                open(config.root_path + '/data/label2id.json',
                     encoding='utf-8'))
            self.ix2label = {v: k for k, v in labelNameToIndex.items()}

    def feature_engineer(self):
        '''
        @description: This function is building all kings of features
        @param {type} None
        @return:
        X_train, feature of train set
        X_test, feature of test set
        y_train, label of train set
        y_test, label of test set
        '''

        logger.info("generate embedding feature ")
        # 获取tfidf 特征， word2vec 特征， word2vec不进行任何聚合
        train_tfidf, train = get_embedding_feature(self.ml_data.train,
                                                   self.ml_data.em.tfidf,
                                                   self.ml_data.em.w2v)
        test_tfidf, test = get_embedding_feature(self.ml_data.dev,
                                                 self.ml_data.em.tfidf,
                                                 self.ml_data.em.w2v)

        logger.info("generate autoencoder feature ")
        # 获取到autoencoder 的embedding, 根据encoder 获取而不是decoder
        train_ae = get_autoencoder_feature(
            train,
            self.ml_data.em.ae.max_features,
            self.ml_data.em.ae.max_len,
            self.ml_data.em.ae.encoder,
            tokenizer=self.ml_data.em.ae.tokenizer)
        test_ae = get_autoencoder_feature(
            test,
            self.ml_data.em.ae.max_features,
            self.ml_data.em.ae.max_len,
            self.ml_data.em.ae.encoder,
            tokenizer=self.ml_data.em.ae.tokenizer)

        logger.info("generate basic feature ")
        # 获取nlp 基本特征
        train = get_basic_feature(train)
        test = get_basic_feature(test)

        logger.info("generate modal feature ")
        # 加载图书封面的文件
        cover = os.listdir(config.root_path + '/data/book_cover/')
        # 根据title 匹配图书封面
        train['cover'] = train['title'].progress_apply(
            lambda x: config.root_path + '/data/book_cover/' + x + '.jpg'
            if x + '.jpg' in cover else '')
        test['cover'] = test['title'].progress_apply(
            lambda x: config.root_path + '/data/book_cover/' + x + '.jpg'
            if x + '.jpg' in cover else '')

        # 根据封面获取封面的embedding
        train['res_embedding'] = train['cover'].progress_apply(
            lambda x: get_img_embedding(x, self.res_model))
        test['res_embedding'] = test['cover'].progress_apply(
            lambda x: get_img_embedding(x, self.res_model))

        train['resnext_embedding'] = train['cover'].progress_apply(
            lambda x: get_img_embedding(x, self.resnext_model))
        test['resnext_embedding'] = test['cover'].progress_apply(
            lambda x: get_img_embedding(x, self.resnext_model))

        train['wide_embedding'] = train['cover'].progress_apply(
            lambda x: get_img_embedding(x, self.wide_model))
        test['wide_embedding'] = test['cover'].progress_apply(
            lambda x: get_img_embedding(x, self.wide_model))

        logger.info("generate bert feature ")
        train['bert_embedding'] = train['text'].progress_apply(
            lambda x: get_pretrain_embedding(x, self.bert_tonkenizer, self.bert
                                             ))
        test['bert_embedding'] = test['text'].progress_apply(
            lambda x: get_pretrain_embedding(x, self.bert_tonkenizer, self.bert
                                             ))

        logger.info("generate lda feature ")
        # 生成bag of word格式数据
        train['bow'] = train['queryCutRMStopWord'].apply(
            lambda x: self.ml_data.em.lda.id2word.doc2bow(x))
        test['bow'] = test['queryCutRMStopWord'].apply(
            lambda x: self.ml_data.em.lda.id2word.doc2bow(x))
        # 在bag of word 基础上得到lda的embedding
        train['lda'] = list(
            map(lambda doc: get_lda_features(self.ml_data.em.lda, doc),
                train['bow']))
        test['lda'] = list(
            map(lambda doc: get_lda_features(self.ml_data.em.lda, doc),
                test['bow']))

        logger.info("formate data")
        #  将所有的特征拼接到一起
        train = formate_data(train, train_tfidf, train_ae)
        test = formate_data(test, test_tfidf, test_ae)
        #  生成训练，测试的数据
        cols = [x for x in train.columns if str(x) not in ['labelIndex']]
        X_train = train[cols]
        X_test = test[cols]
        train["labelIndex"] = train["labelIndex"].astype(int)
        test["labelIndex"] = test["labelIndex"].astype(int)
        y_train = train["labelIndex"]
        y_test = test["labelIndex"]
        return X_train, X_test, y_train, y_test

    def param_search(self, search_method='grid'):
        '''
        @description: use param search tech to find best param
        @param {type}
        search_method: two options. grid or bayesian optimization
        @return: None
        '''
        # 使用网格搜索 或者贝叶斯优化 寻找最优参数
        if search_method == 'grid':
            logger.info("use grid search")
            self.model = Grid_Train_model(self.model, self.X_train,
                                          self.X_test, self.y_train,
                                          self.y_test)
        elif search_method == 'bayesian':
            logger.info("use bayesian optimization")
            trn_data = lgb.Dataset(data=self.X_train,
                                   label=self.y_train,
                                   free_raw_data=False)
            param = bayes_parameter_opt_lgb(trn_data)
            logger.info("best param", param)
            return param

    def unbalance_helper(self,
                         imbalance_method='under_sampling',
                         search_method='grid'):
        '''
        @description: handle unbalance data, then search best param
        @param {type}
        imbalance_method,  three option, under_sampling for ClusterCentroids, SMOTE for over_sampling, ensemble for BalancedBaggingClassifier
        search_method: two options. grid or bayesian optimization
        @return: None
        '''
        logger.info("get all freature")
        # 生成所有feature
        self.X_train, self.X_test, self.y_train, self.y_test = self.feature_engineer(
        )
        model_name = None
        # 是否使用不平衡数据处理方式，上采样， 下采样， ensemble
        if imbalance_method == 'over_sampling':
            logger.info("Use SMOTE deal with unbalance data ")
            self.X_train, self.y_train = SMOTE().fit_resample(
                self.X_train, self.y_train)
            self.X_test, self.y_test = SMOTE().fit_resample(
                self.X_train, self.y_train)
            model_name = 'lgb_over_sampling'
        elif imbalance_method == 'under_sampling':
            logger.info("Use ClusterCentroids deal with unbalance data ")
            self.X_train, self.y_train = ClusterCentroids(
                random_state=0).fit_resample(self.X_train, self.y_train)
            self.X_test, self.y_test = ClusterCentroids(
                random_state=0).fit_resample(self.X_test, self.y_test)
            model_name = 'lgb_under_sampling'
        elif imbalance_method == 'ensemble':
            self.model = BalancedBaggingClassifier(
                base_estimator=DecisionTreeClassifier(),
                sampling_strategy='auto',
                replacement=False,
                random_state=0)
            model_name = 'ensemble'
        logger.info('search best param')
        # 使用set_params 将搜索到的最优参数设置为模型的参数
        if imbalance_method != 'ensemble':
            # param = self.param_search(search_method=search_method)
            # param['params']['num_leaves'] = int(param['params']['num_leaves'])
            # param['params']['max_depth'] = int(param['params']['max_depth'])
            param = {}
            param['params'] = {}
            param['params']['num_leaves'] = 3
            param['params']['max_depth'] = 5
            self.model = self.model.set_params(**param['params'])
        logger.info('fit model ')
        # 训练， 并输出模型的结果
        self.model.fit(self.X_train, self.y_train)
        Test_predict_label = self.model.predict(self.X_test)
        Train_predict_label = self.model.predict(self.X_train)
        per, acc, recall, f1 = get_score(self.y_train, self.y_test,
                                         Train_predict_label,
                                         Test_predict_label)
        # 输出训练集的精确率
        logger.info('Train accuracy %s' % per)
        # 输出测试集的准确率
        logger.info('test accuracy %s' % acc)
        # 输出recall
        logger.info('test recall %s' % recall)
        # 输出F1-score
        logger.info('test F1_score %s' % f1)
        self.save(model_name)

    def process(self, title, desc):
        # 处理数据, 生成模型预测所需要的特征
        df = pd.DataFrame([[title, desc]], columns=['title', 'desc'])
        df['text'] = df['title'] + df['desc']
        df["queryCut"] = df["text"].apply(query_cut)
        df["queryCutRMStopWord"] = df["queryCut"].apply(
            lambda x:
            [word for word in x if word not in self.ml_data.em.stopWords])

        df_tfidf, df = get_embedding_feature(df, self.ml_data.em.tfidf,
                                             self.ml_data.em.w2v)

        print("generate basic feature ")
        df = get_basic_feature(df)

        print("generate modal feature ")
        df['cover'] = ''
        df['res_embedding'] = df.cover.progress_apply(
            lambda x: get_img_embedding(x, self.res_model))

        df['resnext_embedding'] = df.cover.progress_apply(
            lambda x: get_img_embedding(x, self.resnext_model))

        df['wide_embedding'] = df.cover.progress_apply(
            lambda x: get_img_embedding(x, self.wide_model))

        print("generate bert feature ")
        df['bert_embedding'] = df.text.progress_apply(
            lambda x: get_pretrain_embedding(x, self.bert_tonkenizer, self.bert
                                             ))

        print("generate lda feature ")
        df['bow'] = df['queryCutRMStopWord'].apply(
            lambda x: self.ml_data.em.lda.id2word.doc2bow(x))
        df['lda'] = list(
            map(lambda doc: get_lda_features(self.ml_data.em.lda, doc),
                df.bow))

        print("generate autoencoder feature ")
        df_ae = get_autoencoder_feature(df,
                                        self.ml_data.em.ae.max_features,
                                        self.ml_data.em.ae.max_len,
                                        self.ml_data.em.ae.encoder,
                                        tokenizer=self.ml_data.em.ae.tokenizer)

        print("formate data")
        df['labelIndex'] = 1
        df = formate_data(df, df_tfidf, df_ae)
        cols = [x for x in df.columns if str(x) not in ['labelIndex']]
        X_train = df[cols]
        return X_train

    def predict(self, title, desc):
        '''
        @description: 根据输入的title, desc 预测图书的类别
        @param {type}
        title, input
        desc: input
        @return: label
        '''
        inputs = self.process(title, desc)
        label = self.ix2label[self.model.predict(inputs)[0]]
        proba = np.max(self.model.predict_proba(inputs))
        return label, proba

    def save(self, model_name):
        '''
        @description:save model
        @param {type}
        model_name, file name for saving
        @return: None
        '''
        joblib.dump(self.model, root_path + '/model/ml_model/' + model_name)

    def load(self, path):
        '''
        @description: load model
        @param {type}
        path: model path
        @return:None
        '''
        self.model = joblib.load(path)
