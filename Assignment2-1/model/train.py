#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: lpx, jby
@Date: 2020-07-13 12:31:25
@LastEditTime: 2020-07-19 14:07:35
@LastEditors: Please set LastEditors
@Description: Train the model.
@FilePath: /JD_project_2/baseline/model/train.py
'''

import pickle
import os
import sys
import pathlib

import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from tensorboardX import SummaryWriter

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))

from dataset import PairDataset
from model import Seq2seq
import config
from evaluate import evaluate
from dataset import collate_fn, SampleDataset


def train(dataset, val_dataset, v, start_epoch=0):
    """Train the model, evaluate it and store it.

    Args:
        dataset (dataset.PairDataset): The training dataset.
        val_dataset (dataset.PairDataset): The evaluation dataset.
        v (vocab.Vocab): The vocabulary built from the training dataset.
        start_epoch (int, optional): The starting epoch number. Defaults to 0.
    """
    print('loading model')
    DEVICE = torch.device("cuda" if config.is_cuda else "cpu")

    model = Seq2seq(v)
    model.load_model()
    model.to(DEVICE)

    # forward
    print("loading data")
    train_data = SampleDataset(dataset.pairs, v)
    val_data = SampleDataset(val_dataset.pairs, v)

    print("initializing optimizer")

    ###########################################
    #          TODO: module 1 task 2          #
    ###########################################

    # Define the optimizer.
    optimizer = 
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)

    val_losses = np.inf
    if (os.path.exists(config.losses_path)):
        with open(config.losses_path, 'rb') as f:
            val_losses = pickle.load(f)

    ###########################################
    #          TODO: module 3 task 3          #
    ###########################################

    # SummaryWriter: Log writer used for TensorboardX visualization.
    writer = 
    # tqdm: A tool for drawing progress bars during training.
    with tqdm(total=config.epochs) as epoch_progress:
        # Loop for epochs.
        for epoch in range(start_epoch, config.epochs):
            batch_losses = []  # Get loss of each batch.
            with tqdm(total=len(train_dataloader) // config.batch_size)\
                    as batch_progress:
                # Lopp for batches.
                for batch, data in enumerate(tqdm(train_dataloader)):
                    x, y, x_len, y_len, oov, len_oovs = data
                    assert not np.any(np.isnan(x.numpy()))
                    if config.is_cuda:  # Training with GPUs.
                        x = x.to(DEVICE)
                        y = y.to(DEVICE)
                        x_len = x_len.to(DEVICE)
                        len_oovs = len_oovs.to(DEVICE)

                    ###########################################
                    #          TODO: module 3 task 1          #
                    ###########################################


                    ###########################################
                    #          TODO: module 3 task 2          #
                    ###########################################

                    # Do gradient clipping to prevent gradient explosion.

                    # Update weights.

                    # Output and record epoch loss every 100 batches.
                    if (batch % 100) == 0:
                        batch_progress.set_description(f'Epoch {epoch}')
                        batch_progress.set_postfix(Batch=batch,
                                                   Loss=loss.item())
                        batch_progress.update()
                        ###########################################
                        #          TODO: module 3 task 3          #
                        ###########################################

                        # Write loss for tensorboard.

            # Calculate average loss over all batches in an epoch.
            epoch_loss = np.mean(batch_losses)

            epoch_progress.set_description(f'Epoch {epoch}')
            epoch_progress.set_postfix(Loss=epoch_loss)
            epoch_progress.update()
            # Calculate evaluation loss.
            avg_val_loss = evaluate(model, val_data, epoch)

            print('training loss:{}'.format(epoch_loss),
                  'validation loss:{}'.format(avg_val_loss))

            # Update minimum evaluating loss.
            if (avg_val_loss < val_losses):
                torch.save(model.encoder, config.encoder_save_name)
                torch.save(model.decoder, config.decoder_save_name)
                torch.save(model.attention, config.attention_save_name)
                torch.save(model.reduce_state, config.reduce_state_save_name)
                val_losses = avg_val_loss
            with open(config.losses_path, 'wb') as f:
                pickle.dump(val_losses, f)

    writer.close()


if __name__ == "__main__":
    # Prepare dataset for training.
    DEVICE = torch.device('cuda') if config.is_cuda else torch.device('cpu')
    dataset = PairDataset(config.data_path,
                          max_src_len=config.max_src_len,
                          max_tgt_len=config.max_tgt_len,
                          truncate_src=config.truncate_src,
                          truncate_tgt=config.truncate_tgt)
    val_dataset = PairDataset(config.val_data_path,
                              max_src_len=config.max_src_len,
                              max_tgt_len=config.max_tgt_len,
                              truncate_src=config.truncate_src,
                              truncate_tgt=config.truncate_tgt)

    vocab = dataset.build_vocab(embed_file=config.embed_file)

    train(dataset, val_dataset, vocab, start_epoch=0)
