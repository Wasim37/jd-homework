#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Please set LastEditors
Date: 2020-09-11 11:44:54
LastEditTime: 2020-10-16 16:30:31
FilePath: /Assignment3-2_solution/ranking/train_matchnn.py
Desciption: Train a matching network.
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''

import os
import torch
from torch.utils.data import DataLoader
from data import DataPrecessForSentence
from matchnn_utils import train, validate
from transformers import BertTokenizer
from matchnn import BertModelTrain
from transformers.optimization import AdamW
import sys
sys.path.append('..')
from config import is_cuda, root_path, max_sequence_length

seed = 9
torch.manual_seed(seed)
if is_cuda:
    torch.cuda.manual_seed_all(seed)


def main(train_file,
         dev_file,
         target_dir,
         epochs=10,
         batch_size=32,
         lr=2e-05,
         patience=3,
         max_grad_norm=10.0,
         checkpoint=None):
    """训练一个BERT模型对输入的两个问题做序列相似度的匹配，得到相似度分数

    Args:
        train_file ([type]): [description]
        dev_file ([type]): [description]
        target_dir ([type]): [description]
        epochs (int, optional): [description]. Defaults to 10.
        batch_size (int, optional): [description]. Defaults to 32.
        lr ([type], optional): [description]. Defaults to 2e-05.
        patience (int, optional): [description]. Defaults to 3.
        max_grad_norm (float, optional): [description]. Defaults to 10.0.
        checkpoint ([type], optional): [description]. Defaults to None.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    device = torch.device("cuda") if is_cuda else torch.device("cpu")

    bert_tokenizer = BertTokenizer.from_pretrained(os.path.join(root_path,
                                                   'lib/bert/vocab.txt'),
                                                   do_lower_case=True)
    print(20 * "=", " Preparing for training ", 20 * "=")

    # 加载数据
    print("\t* Loading training data...")
    train_data = DataPrecessForSentence(bert_tokenizer, train_file)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    print("\t* Loading validation data...")
    dev_data = DataPrecessForSentence(bert_tokenizer, dev_file)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)

    # 定义模型
    print("\t* Building model...")
    model = BertModelTrain().to(device)
    # 返回model的所有参数的(name, tensor)的键值对，并更改网络参数，进行 fine-tuning
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01
    }, {
        'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
    }]
    # AdamW是实现了权重衰减的优化器
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    # 学习率调整：https://blog.csdn.net/weixin_40100431/article/details/84311430
    # mode: 可选择‘min’或者‘max’，min表示当监控量停止下降的时候，学习率将减小，max表示当监控量停止上升的时候，学习率将减小。默认值为‘min’
    # factor: 学习率每次降低多少，new_lr = old_lr * factor
    # patience: 容忍网路的性能不提升的次数，高于这个次数就降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.85,
                                                           patience=0)
    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    valid_losses = []
    # Continuing training from a checkpoint if one was given as argument
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        print("\t* Training will continue on existing model from epoch {}...".format(start_epoch))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]
    # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy, auc = validate(model, dev_loader)
    print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}"
          .format(valid_loss, (valid_accuracy * 100), auc))

    # 开始训练 ...
    print("\n", 20 * "=", "Training Bert model on device: {}".format(device), 20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)
        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader,
                                                       optimizer, epoch,
                                                       max_grad_norm)
        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%".
              format(epoch_time, epoch_loss, (epoch_accuracy * 100)))
        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy, epoch_auc = validate(model, dev_loader)
        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100), epoch_auc))
        
        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)
        
        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
            if patience_counter >= patience:
                print("-> Early stopping: patience limit reached, stopping...")
                break
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_score": best_score,
                    "epochs_count": epochs_count,
                    "train_losses": train_losses,
                    "valid_losses": valid_losses
                }, os.path.join(target_dir, "best.pth.tar"))


if __name__ == "__main__":
    main(os.path.join(root_path, 'data/ranking/train.tsv'),
         os.path.join(root_path, 'data/ranking/dev.tsv'),
         os.path.join(root_path, "model/ranking/"))
