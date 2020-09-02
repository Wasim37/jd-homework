#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: your name
@Date: 2020-07-13 11:00:51
@LastEditTime: 2020-07-18 17:24:43
@LastEditors: Please set LastEditors
@Description: Define the model.
@FilePath: /JD_project_2/baseline/model/model.py
'''


import os
import sys
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))
import config


class Encoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 hidden_size,
                 rnn_drop: float = 0):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embed_size,
                            hidden_size,
                            bidirectional=True,
                            dropout=rnn_drop,
                            batch_first=True)

    def forward(self, x):
        """Define forward propagation for the endoer.

        Args:
            x (Tensor): The input samples as shape (batch_size, seq_len).

        Returns:
            output (Tensor):
                The output of lstm with shape
                (batch_size, seq_len, 2 * hidden_units).
            hidden (tuple):
                The hidden states of lstm (h_n, c_n).
                Each with shape (2, batch_size, hidden_units)
        """
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded)

        return output, hidden


class Attention(nn.Module):
    def __init__(self, hidden_units):
        super(Attention, self).__init__()
        # Define feed-forward layers.
        self.Wh = nn.Linear(2*hidden_units, 2*hidden_units, bias=False)
        self.Ws = nn.Linear(2*hidden_units, 2*hidden_units)
        self.v = nn.Linear(2*hidden_units, 1, bias=False)

    def forward(self, decoder_states, encoder_output, x_padding_masks):
        """Define forward propagation for the attention network.

        Args:
            decoder_states (tuple):
                The hidden states from lstm (h_n, c_n) in the decoder,
                each with shape (1, batch_size, hidden_units)
            encoder_output (Tensor):
                The output from the lstm in the decoder with
                shape (batch_size, seq_len, hidden_units).
            x_padding_masks (Tensor):
                The padding masks for the input sequences
                with shape (batch_size, seq_len).

        Returns:
            context_vector (Tensor):
                Dot products of attention weights and encoder hidden states.
                The shape is (batch_size, 2*hidden_units).
            attention_weights (Tensor): The shape is (batch_size, seq_length).
        """
        # Concatenate h and c to get s_t and expand the dim of s_t.
        h_dec, c_dec = decoder_states
        # (1, batch_size, 2*hidden_units)
        s_t = torch.cat([h_dec, c_dec], dim=2)
        # (batch_size, 1, 2*hidden_units)
        s_t = s_t.transpose(0, 1)
        # (batch_size, seq_length, 2*hidden_units)
        s_t = s_t.expand_as(encoder_output).contiguous()

        # calculate attention scores
        # Equation(11).
        # Wh h_* (batch_size, seq_length, 2*hidden_units)
        encoder_features = self.Wh(encoder_output.contiguous())
        # Ws s_t (batch_size, seq_length, 2*hidden_units)
        decoder_features = self.Ws(s_t)
        # (batch_size, seq_length, 2*hidden_units)
        att_inputs = encoder_features + decoder_features
        # (batch_size, seq_length, 1)
        score = self.v(torch.tanh(att_inputs))
        # (batch_size, seq_length)
        attention_weights = F.softmax(score, dim=1).squeeze(2)
        attention_weights = attention_weights * x_padding_masks
        # Normalize attention weights after excluding padded positions.
        normalization_factor = attention_weights.sum(1, keepdim=True)
        attention_weights = attention_weights / normalization_factor
        # (batch_size, 1, 2*hidden_units)
        context_vector = torch.bmm(attention_weights.unsqueeze(1),
                                   encoder_output)
        # (batch_size, 2*hidden_units)
        context_vector = context_vector.squeeze(1)

        return context_vector, attention_weights


class Decoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 hidden_size,
                 enc_hidden_size=None,
                 is_cuda=True):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.DEVICE = torch.device('cuda') if is_cuda else torch.device('cpu')
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

        self.W1 = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.W2 = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, decoder_input, decoder_states, encoder_output,
                context_vector):
        """Define forward propagation for the decoder.

        Args:
            decoder_input (Tensor):
                The input of the decoder x_t of shape (batch_size, 1).
            decoder_states (tuple):
                The hidden states(h_n, c_n) of the decoder from last time step.
                The shapes are (1, batch_size, hidden_units) for each.
            encoder_output (Tensor):
                The output from the encoder of shape
                (batch_size, seq_length, 2*hidden_units).
            context_vector (Tensor):
                The context vector from the attention network
                of shape (batch_size,2*hidden_units).

        Returns:
            p_vocab (Tensor):
                The vocabulary distribution of shape (batch_size, vocab_size).
            docoder_states (tuple):
                The lstm states in the decoder.
                The shapes are (1, batch_size, hidden_units) for each.
        """
        decoder_emb = self.embedding(decoder_input)

        decoder_output, decoder_states = self.lstm(decoder_emb, decoder_states)

        # concatenate context vector and decoder state
        # (batch_size, 3*hidden_units)
        decoder_output = decoder_output.view(-1, config.hidden_size)
        concat_vector = torch.cat(
            [decoder_output,
             context_vector],
            dim=-1)

        # calculate vocabulary distribution
        # (batch_size, hidden_units)
        FF1_out = self.W1(concat_vector)
        # (batch_size, vocab_size)
        FF2_out = self.W2(FF1_out)
        # (batch_size, vocab_size)
        p_vocab = F.softmax(FF2_out, dim=1)

        return p_vocab, decoder_states


class ReduceState(nn.Module):
    """
    Since the encoder has a bidirectional LSTM layer while the decoder has a
    unidirectional LSTM layer, we add this module to reduce the hidden states
    output by the encoder (merge two directions) before input the hidden states
    nto the decoder.
    """
    def __init__(self):
        super(ReduceState, self).__init__()

    def forward(self, hidden):
        """The forward propagation of reduce state module.

        Args:
            hidden (tuple):
                Hidden states of encoder,
                each with shape (2, batch_size, hidden_units).

        Returns:
            tuple:
                Reduced hidden states,
                each with shape (1, batch_size, hidden_units).
        """
        h, c = hidden
        h_reduced = torch.sum(h, dim=0, keepdim=True)
        c_reduced = torch.sum(c, dim=0, keepdim=True)
        return (h_reduced, c_reduced)


class Seq2seq(nn.Module):
    def __init__(
            self,
            v
    ):
        super(Seq2seq, self).__init__()
        self.v = v
        self.DEVICE = torch.device("cuda" if config.is_cuda else "cpu")
        self.attention = Attention(config.hidden_size)
        self.encoder = Encoder(
            len(v),
            config.embed_size,
            config.hidden_size,
        )
        self.decoder = Decoder(len(v),
                               config.embed_size,
                               config.hidden_size,
                               )
        self.reduce_state = ReduceState()

    def load_model(self):
        if (os.path.exists(config.encoder_save_name)):
            self.encoder = torch.load(config.encoder_save_name)
            self.decoder = torch.load(config.decoder_save_name)
            self.attention = torch.load(config.attention_save_name)
            self.reduce_state = torch.load(config.reduce_state_save_name)

    def forward(self, x, x_len, y, len_oovs, batch):
        """Define the forward propagation for the seq2seq model.

        Args:
            x (Tensor):
                Input sequences as source with shape (batch_size, seq_len)
            x_len ([int): Sequence length of the current batch.
            y (Tensor):
                Input sequences as reference with shape (bacth_size, y_len)
            len_oovs (int):
                The number of out-of-vocabulary words in this sample.
            batch (int): The number of the current batch.

        Returns:
            batch_loss (Tensor): The average loss of the current batch.
        """
        oov_token = torch.full(x.shape, self.v.UNK).long().to(self.DEVICE)
        x_copy = torch.where(x > len(self.v) - 1, oov_token, x)
        x_padding_masks = torch.ne(x_copy, 0).byte().float()
        encoder_output, encoder_states = self.encoder(x_copy)
        # Reduce encoder hidden states.
        decoder_states = self.reduce_state(encoder_states)

        # Calculate loss for every step.
        step_losses = []
        for t in range(y.shape[1]-1):
            decoder_input_t = y[:, t]  # x_t
            decoder_target_t = y[:, t+1]  # y_t
            # Get context vector from the attention network.
            context_vector, attention_weights = self.attention(
                decoder_states, encoder_output, x_padding_masks)
            # Get vocab distribution and hidden states from the decoder.
            p_vocab, decoder_states = self.decoder(
                decoder_input_t.unsqueeze(1), decoder_states, encoder_output,
                context_vector)

            # Get the probabilities predict by the model for target tokens.
            target_probs = torch.gather(p_vocab,
                                        1,
                                        decoder_target_t.unsqueeze(1))
            target_probs = target_probs.squeeze(1)
            # Apply a mask such that pad zeros do not affect the loss
            mask = torch.ne(decoder_target_t, 0).byte()
            # Do smoothing to prevent getting NaN loss because of log(0).
            loss = -torch.log(target_probs+config.eps)
            mask = mask.float()
            loss = loss * mask
            step_losses.append(loss)

        sample_losses = torch.sum(torch.stack(step_losses, 1), 1)
        # get the non-padded length of each sequence in the batch
        seq_len_mask = torch.ne(y, 0).byte().float()
        batch_seq_len = torch.sum(seq_len_mask, dim=1)

        # get batch loss by dividing the loss of each batch
        # by the target sequence length and mean
        batch_loss = torch.mean(sample_losses / batch_seq_len)
        return batch_loss
