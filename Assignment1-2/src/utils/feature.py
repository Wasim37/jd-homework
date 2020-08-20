'''
@Author: xiaoyao jiang
LastEditors: xiaoyao jiang
@Date: 2020-07-01 11:35:56
LastEditTime: 2020-08-13 18:37:31
FilePath: /bookClassification/src/utils/feature.py
@Desciption:  feature engineering
'''

import numpy as np
import copy
from __init__ import *
from src.utils.tools import wam, format_data
from src.utils import config
import pandas as pd
import joblib
import json
import string
import jieba.posseg as pseg
from PIL import Image
import torchvision.transforms as transforms


def get_autoencoder_feature(data, max_features, max_len, model,
                            tokenizer=None):
    '''
    @description: 获取autoencoder 特征
    @param {type}
    train, train data set
    test, test data set
    max_features, max_features
    max_len, max_len
    model, autoencoder model
    tokenizer, autoencoder tokenizer
    @return: DataFrame of train and test
    '''
    # 格式化数据
    X, _ = format_data(data,
                       max_features,
                       max_len,
                       tokenizer=tokenizer,
                       shuffle=True)
    # 使用autoencoder 的encoder 进行预测
    data_ae = pd.DataFrame(model.predict(X, batch_size=64, verbose=1).max(axis=1),
                           columns=['ae' + str(i) for i in range(max_len)])
    return data_ae


def get_lda_features(lda_model, document):
    '''
    @description: Transforms a bag of words document to features.
    It returns the proportion of how much each topic was
    present in the document.
    @param {type}
    lda_model: lda_model
    document, input
    @return: lda feature
    '''
    # 基于bag of word 格式数据获取lda的特征
    topic_importances = lda_model.get_document_topics(document,
                                                      minimum_probability=0)
    topic_importances = np.array(topic_importances)
    return topic_importances[:, 1]


def get_pretrain_embedding(text, tokenizer, model):
    '''
    @description:  get bert embedding
    @param {type}
    text: input
    tokenizer, bert tokenizer
    model, bert model
    @return: bert embedding ndarray
    '''
    # 通过bert tokenizer 来处理数据， 然后使用bert model 获取bert embedding
    text_dict = tokenizer.encode_plus(
        text,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=400,  # Pad & truncate all sentences.
        ad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',
    )
    input_ids, attention_mask, token_type_ids = text_dict[
        'input_ids'], text_dict['attention_mask'], text_dict['token_type_ids']
    _, res = model(input_ids.to(config.device),
                   attention_mask=attention_mask.to(config.device),
                   token_type_ids=token_type_ids.to(config.device))
    return res.detach().cpu().numpy()[0]


def get_transforms():
    '''
    @description: transform image data
    @param {type}None
    @return:transformed data
    '''
    # 将图片数据处理为统一格式
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.46777044, 0.44531429, 0.40661017],
            std=[0.12221994, 0.12145835, 0.14380469],
        ),
    ])


def get_img_embedding(cover, model):
    '''
    @description: get_img_embedding
    @param {type}
    cover,  book's cover image path
    model, image network model
    @return: modal feature
    '''
    # 处理图片数据， 传入的不是图片则 生成（1， 1000）的0向量
    transforms = get_transforms()
    if str(cover)[-3:] != 'jpg':
        return np.zeros((1, 1000))[0]
    image = Image.open(cover).convert("RGB")
    image = transforms(image).to(config.device)
    return model(image.unsqueeze(0)).detach().cpu().numpy()[0]


def get_embedding_feature(data, tfidf, embedding_model):
    '''
    @description: get_embedding_feature, tfidf, word2vec -> max/mean, word2vec n-gram(2, 3, 4) -> max/mean, label embedding->max/mean
    @param {type}
    mldata, input data set, mldata class instance
    @return:
    train_tfidf, tfidf of train data set
    test_tfidf, tfidf of test data set
    train, train data set
    test, test data set
    '''
    # 根据过滤停止词后的数据， 获取tfidf 特征
    data["queryCutRMStopWords"] = data["queryCutRMStopWord"].apply(
        lambda x: " ".join(x))
    tfidf_data = pd.DataFrame(
        tfidf.transform(data["queryCutRMStopWords"].tolist()).toarray())
    tfidf_data.columns = ['tfidf' + str(i) for i in range(tfidf_data.shape[1])]

    print("transform w2v")
    # 同上， 获取embedding 特征， 不进行聚合
    data['w2v'] = data["queryCutRMStopWord"].apply(
        lambda x: wam(x, embedding_model, aggregate=False))  # [seq_len * 300]

    # 深度拷贝数据
    train = copy.deepcopy(data)
    # 加载所有类别， 获取类别的embedding， 并保存文件
    labelNameToIndex = json.load(
        open(config.root_path + '/data/label2id.json', encoding='utf-8'))
    labelIndexToName = {v: k for k, v in labelNameToIndex.items()}
    w2v_label_embedding = np.array([
        embedding_model.wv.get_vector(labelIndexToName[key])
        for key in labelIndexToName
        if labelIndexToName[key] in embedding_model.wv.vocab.keys()
    ])

    joblib.dump(w2v_label_embedding,
                config.root_path + '/data/w2v_label_embedding.pkl')
    # 根据未聚合的embedding 数据， 获取各类embedding 特征
    train = generate_feature(train, w2v_label_embedding, model_name='w2v')
    return tfidf_data, train


ch2en = {
    '！': '!',
    '？': '?',
    '｡': '.',
    '（': '(',
    '）': ')',
    '，': ',',
    '：': ':',
    '；': ';',
    '｀': ','
}


def tag_part_of_speech(data):
    '''
    @description: tag part of speech, then calculate the num of noun, adj and verb
    @param {type}
    data, input data
    @return:
    noun_count,num of noun
    adjective_count, num of adj
    verb_count, num of verb
    '''
    # 获取文本的词性， 并计算名词，动词， 形容词的个数
    words = [tuple(x) for x in list(pseg.cut(data))]
    noun_count = len(
        [w for w in words if w[1] in ('NN', 'NNP', 'NNPS', 'NNS')])
    adjective_count = len([w for w in words if w[1] in ('JJ', 'JJR', 'JJS')])
    verb_count = len([
        w for w in words if w[1] in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')
    ])
    return noun_count, adjective_count, verb_count


def get_basic_feature(df):
    '''
    @description: get_basic_feature, length, capitals number, num_exclamation_marks, num_punctuation, num_question_marks, num_words, num_unique_words .etc
    @param {type}
    df, dataframe
    @return:
    df, dataframe
    '''
    # 将title 和 desc 拼接
    df['text'] = df['title'] + df['desc']
    # 分词
    df['queryCut'] = df['queryCut'].progress_apply(
        lambda x: [i if i not in ch2en.keys() else ch2en[i] for i in x])
    # 文本的长度
    df['length'] = df['queryCut'].progress_apply(lambda x: len(x))
    # 大写的个数
    df['capitals'] = df['queryCut'].progress_apply(
        lambda x: sum(1 for c in x if c.isupper()))
    # 大写 与 文本长度的占比
    df['caps_vs_length'] = df.progress_apply(
        lambda row: float(row['capitals']) / float(row['length']), axis=1)
    # 感叹号的个数
    df['num_exclamation_marks'] = df['queryCut'].progress_apply(
        lambda x: x.count('!'))
    # 问号个数
    df['num_question_marks'] = df['queryCut'].progress_apply(
        lambda x: x.count('?'))
    # 标点符号个数
    df['num_punctuation'] = df['queryCut'].progress_apply(
        lambda x: sum(x.count(w) for w in string.punctuation))
    # *&$%字符的个数
    df['num_symbols'] = df['queryCut'].progress_apply(
        lambda x: sum(x.count(w) for w in '*&$%'))
    # 词的个数
    df['num_words'] = df['queryCut'].progress_apply(lambda x: len(x))
    # 唯一词的个数
    df['num_unique_words'] = df['queryCut'].progress_apply(
        lambda x: len(set(w for w in x)))
    # 唯一词 与总词数的比例
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
    # 获取名词， 形容词， 动词的个数， 使用tag_part_of_speech函数
    df['nouns'], df['adjectives'], df['verbs'] = zip(
        *df['text'].progress_apply(lambda x: tag_part_of_speech(x)))
    # 名词占总长度的比率
    df['nouns_vs_length'] = df['nouns'] / df['length']
    # 形容词占总长度的比率
    df['adjectives_vs_length'] = df['adjectives'] / df['length']
    # 动词占总长度的比率
    df['verbs_vs_length'] = df['verbs'] / df['length']
    # 名词占总词数的比率
    df['nouns_vs_words'] = df['nouns'] / df['num_words']
    # 形容词占总词数的比率
    df['adjectives_vs_words'] = df['adjectives'] / df['num_words']
    # 动词占总词数的比率
    df['verbs_vs_words'] = df['verbs'] / df['num_words']
    # 首字母大写其他小写的个数
    df["count_words_title"] = df["queryCut"].progress_apply(
        lambda x: len([w for w in x if w.istitle()]))
    # 平均词的个数
    df["mean_word_len"] = df["text"].progress_apply(
        lambda x: np.mean([len(w) for w in x]))
    # 标点符号的占比
    df['punct_percent'] = df['num_punctuation'] * 100 / df['num_words']
    return df


def generate_feature(data, label_embedding, model_name='w2v'):
    '''
    @description: word2vec -> max/mean, word2vec n-gram(2, 3, 4) -> max/mean, label embedding->max/mean
    @param {type}
    data， input data, DataFrame
    label_embedding, all label embedding
    model_name, w2v means word2vec
    @return: data, DataFrame
    '''
    print('generate w2v & fast label max/mean')
    # 首先在预训练的词向量中获取标签的词向量句子,每一行表示一个标签表示
    # 每一行表示一个标签的embedding
    # 计算label embedding 具体参见文档
    data[model_name + '_label_mean'] = data[model_name].progress_apply(
        lambda x: Find_Label_embedding(x, label_embedding, method='mean'))
    data[model_name + '_label_max'] = data[model_name].progress_apply(
        lambda x: Find_Label_embedding(x, label_embedding, method='max'))

    print('generate embedding max/mean')
    # 将embedding 进行max, mean聚合
    data[model_name + '_mean'] = data[model_name].progress_apply(
        lambda x: np.mean(np.array(x), axis=0))
    data[model_name + '_max'] = data[model_name].progress_apply(
        lambda x: np.max(np.array(x), axis=0))

    print('generate embedding window max/mean')
    # 滑窗处理embedding 然后聚合
    data[model_name + '_win_2_mean'] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 2, method='mean'))
    data[model_name + '_win_3_mean'] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 3, method='mean'))
    data[model_name + '_win_4_mean'] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 4, method='mean'))
    data[model_name + '_win_2_max'] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 2, method='max'))
    data[model_name + '_win_3_max'] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 3, method='max'))
    data[model_name + '_win_4_max'] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 4, method='max'))
    return data


def Find_embedding_with_windows(embedding_matrix, window_size=2,
                                method='mean'):
    '''
    @description: generate embedding use window
    @param {type}
    embedding_matrix, input sentence's embedding
    window_size, 2, 3, 4
    method, max/ mean
    @return: ndarray of embedding
    '''
    # 最终的词向量
    result_list = []
    # 遍历input的长度， 根据窗口的大小获取embedding， 进行mean操作， 然后将得到的结果extend到list中， 最后进行mean max 聚合
    for k1 in range(len(embedding_matrix)):
        # 如何当前位置 + 窗口大小 超过input的长度， 则取当前位置到结尾
        # mean 操作后要reshape 为 （1， 300）大小
        if int(k1 + window_size) > len(embedding_matrix):
            result_list.extend(
                np.mean(embedding_matrix[k1:], axis=0).reshape(1, 300))
        else:
            result_list.extend(
                np.mean(embedding_matrix[k1:k1 + window_size],
                        axis=0).reshape(1, 300))
    if method == 'mean':
        return np.mean(result_list, axis=0)
    else:
        return np.max(result_list, axis=0)


def softmax(x):
    '''
    @description: calculate softmax
    @param {type}
    x, ndarray of embedding
    @return: softmax result
    '''
    return np.exp(x) / np.exp(x).sum(axis=0)


def Find_Label_embedding(example_matrix, label_embedding, method='mean'):
    '''
    @description: 根据论文《Joint embedding of words and labels》获取标签空间的词嵌入
    @param {type}
    example_matrix(np.array 2D): denotes words embedding of input
    label_embedding(np.array 2D): denotes the embedding of all label
    @return: (np.array 1D) the embedding by join label and word
    '''

    # 根据矩阵乘法来计算label与word之间的相似度
    similarity_matrix = np.dot(example_matrix, label_embedding.T) / (
        np.linalg.norm(example_matrix) * (np.linalg.norm(label_embedding)))

    # 然后对相似矩阵进行均值池化，则得到了“类别-词语”的注意力机制
    # 这里可以使用max-pooling和mean-pooling
    attention = similarity_matrix.max()
    attention = softmax(attention)
    # 将样本的词嵌入与注意力机制相乘得到
    attention_embedding = example_matrix * attention
    if method == 'mean':
        return np.mean(attention_embedding, axis=0)
    else:
        return np.max(attention_embedding, axis=0)
