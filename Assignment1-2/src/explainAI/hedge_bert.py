'''
@Author: xiaoyao jiang
@LastEditors: xiaoyao jiang
@Date: 2020-07-01 17:57:53
@LastEditTime: 2020-07-13 10:53:09
@FilePath: /bookClassification/src/explainAI/hedge_bert.py
@Desciption:  from https://github.com/UVa-NLP/HEDGE/blob/master/bert/hedge_main_bert_imdb.py
'''
from __future__ import absolute_import, division, print_function

import itertools
import logging
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from __init__ import *
from src.data.dataset import MyDataset, collate_fn
from src.utils import config
from src.DL.models.bert import Model
from transformers import BertTokenizer

import hedge

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)


def set_seed(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)


def evaluate(config,
             model,
             tokenizer,
             eval_dataset,
             fileobject,
             start_pos=0,
             end_pos=2000):

    config.eval_batch_size = config.per_gpu_eval_batch_size
    # Note that DistributedSampler samples randomly
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 drop_last=True,
                                 collate_fn=collate_fn)

    # Eval!
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    count = start_pos
    for batch in itertools.islice(eval_dataloader, start_pos, end_pos):
        start_time = time.time()
        #     for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(config.device) for t in batch)
        print(batch)
        count += 1
        fileobject.write(str(count))
        fileobject.write('\n')
        ori_text_idx = list(batch[0].cpu().numpy()[0])
        if 0 in ori_text_idx:
            ori_text_idx = [idx for idx in ori_text_idx if idx != 0]
        pad_start = len(ori_text_idx)

        with torch.no_grad():
            inputs = [
                torch.unsqueeze(batch[0][0, :pad_start], 0),  # input_ids
                torch.unsqueeze(batch[1][0, :pad_start], 0),  # attention_mask
                torch.unsqueeze(batch[2][0, :pad_start], 0)  # token_type_ids
            ]
            outputs = model(inputs)
            logits = outputs[:2]
        nb_eval_steps += 1

        print(count, len(inputs[0][0]) - 2)
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = batch[3].detach().cpu().numpy()  # label
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids,
                batch[3].detach().cpu().numpy(),  # label
                axis=0)

        for btxt in ori_text_idx:
            if (tokenizer.ids_to_tokens[btxt] != '[CLS]'
                    and tokenizer.ids_to_tokens[btxt] != '[SEP]'):
                fileobject.write(tokenizer.ids_to_tokens[btxt])
                fileobject.write(' ')
        fileobject.write(' >> ')
        if batch[3].cpu().numpy()[0] == 0:
            fileobject.write('0')
            fileobject.write(' ||| ')
        else:
            fileobject.write('1')
            fileobject.write(' ||| ')
        print('HEDGE')
        shap = hedge.HEDGE(model, inputs, config, thre=100)
        print('shap', shap)
        shap.compute_shapley_hier_tree(model, inputs, 2)
        word_list, _ = shap.get_importance_phrase()
        print('word_list', word_list)

        for feaidx in word_list:
            if len(feaidx) == 1:
                if tokenizer.ids_to_tokens[ori_text_idx[
                        feaidx[0]]] != '[CLS]' and tokenizer.ids_to_tokens[
                            ori_text_idx[feaidx[0]]] != '[SEP]':
                    fileobject.write(str(feaidx[0]))
                    fileobject.write(' ')
            else:
                fea_end = -1
                for fea in feaidx[-1::-1]:
                    if tokenizer.ids_to_tokens[ori_text_idx[
                            fea]] != '[CLS]' and tokenizer.ids_to_tokens[
                                ori_text_idx[fea]] != '[SEP]':
                        fea_end = fea
                        break
                if fea_end > -1 and fea_end > feaidx[0]:
                    fileobject.write(str(feaidx[0]))
                    fileobject.write('-')
                    fileobject.write(str(fea_end))
                    fileobject.write(' ')

        fileobject.write(' >> ')
        if np.argmax(logits.detach().cpu().numpy(), axis=1) == 0:
            fileobject.write('0')
        else:
            fileobject.write('1')
        fileobject.write('\n')
        end_time = time.time()
        print('Elasped time: {}'.format(end_time - start_time))

    preds = np.argmax(preds, axis=1)
    eval_acc = (preds == out_label_ids).mean()

    return eval_acc


if __name__ == '__main__':
    # Set seed
    set_seed(config)

    config.bert_path = config.root_path + '/model/bert/'
    config.hidden_size = 768
    model = Model(config).to(config.device)

    checkpoint = torch.load(config.root_path + '/model/saved_dict/bert.ckpt')
    model.load_state_dict(checkpoint, strict=False)
    tokenizer = BertTokenizer.from_pretrained(
        config.root_path + '/model/bert', do_lower_case=config.do_lower_case)
    model.to(config.device)
    print('finish model load')
    if config.visualize > -1:
        start_pos = config.visualize
        end_pos = start_pos + 1
    else:
        start_pos = config.start_pos
        end_pos = config.end_pos
    print('load data')
    test_dataset = MyDataset(config.test_file,
                             None,
                             config.max_length,
                             tokenizer=tokenizer,
                             word=False)
    file_name = 'hedge_bert_' + str(config.start_pos) + '-' + str(
        config.end_pos) + '.txt'
    with open(file_name, 'w') as f:
        print('evaluate')
        test_acc = evaluate(config, model, tokenizer, test_dataset, f,
                            start_pos, end_pos)
    print('\ntest_acc {:.6f}'.format(test_acc))
