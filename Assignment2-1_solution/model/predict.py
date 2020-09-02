#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: lpx, jby
@Date: 2020-07-13 11:00:51
@LastEditTime: 2020-07-21 11:30:41
@LastEditors: Please set LastEditors
@Description: Generate a summary.
@FilePath: /JD_project_2/baseline/model/predict.py
'''

import random
import os
import sys
import pathlib

import torch
import jieba

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))

import config
from model import Seq2seq
from dataset import PairDataset
from utils import source2ids, outputids2words, Beam, timer, add2heap


class Predict():
    @timer(module='initalize predicter')
    def __init__(self):
        self.DEVICE = torch.device("cuda" if config.is_cuda else "cpu")
        dataset = PairDataset(config.data_path,
                              max_src_len=config.max_src_len,
                              max_tgt_len=config.max_tgt_len,
                              truncate_src=config.truncate_src,
                              truncate_tgt=config.truncate_tgt)

        self.vocab = dataset.build_vocab(embed_file=config.embed_file)

        self.model = Seq2seq(self.vocab)
        self.stop_word = list(
            set([
                self.vocab[x.strip()] for x in
                open(config.stop_word_file
                     ).readlines()
            ]))
        self.model.load_model()
        self.model.to(self.DEVICE)

    def greedy_search(self,
                      encoder_input,
                      max_sum_len,
                      max_oovs,
                      x_padding_masks):
        """Function which returns a summary by always picking
           the highest probability option conditioned on the previous word.

        Args:
            encoder_input (Tensor): Input sequence as the source.
            max_sum_len (int): The maximum length a summary can have.
            max_oovs (int): Number of out-of-vocabulary tokens.

        Returns:
            summary (list): The token list of the result summary.
        """

        # Get encoder output and states.
        encoder_output, encoder_states = self.model.encoder(encoder_input)

        # Initialize decoder's hidden states with encoder's hidden states.
        decoder_states = self.model.reduce_state(encoder_states)

        # Initialize decoder's input at time step 0 with the SOS token.
        decoder_input_t = torch.ones(1) * self.vocab.SOS
        decoder_input_t = decoder_input_t.to(self.DEVICE, dtype=torch.int64)
        summary = [self.vocab.SOS]

        # Generate hypothesis with maximum decode step.
        while int(decoder_input_t.item()) != (self.vocab.EOS) \
                and len(summary) < max_sum_len:

            context_vector, attention_weights = \
                self.model.attention(decoder_states,
                                     encoder_output,
                                     x_padding_masks)
            p_vocab, decoder_states = \
                self.model.decoder(decoder_input_t.unsqueeze(1),
                                   decoder_states,
                                   encoder_output,
                                   context_vector)
            # Get next token with maximum probability.
            decoder_input_t = torch.argmax(p_vocab, dim=1).to(self.DEVICE)
            decoder_word_idx = decoder_input_t.item()
            summary.append(decoder_word_idx)
            decoder_input_t = self.replace_oov(decoder_input_t)

        return summary

#     @timer('best k')
    def best_k(self, beam, k, encoder_output, x_padding_masks):
        """Get best k tokens to extend the current sequence at the current time step.

        Args:
            beam (untils.Beam): The candidate beam to be extended.
            k (int): Beam size.
            encoder_output (Tensor): The lstm output from the encoder.
            x_padding_masks (Tensor):
                The padding masks for the input sequences.

        Returns:
            best_k (list(Beam)): The list of best k candidates.

        """
        # use decoder to generate vocab distribution for the next token
        decoder_input_t = torch.tensor(beam.tokens[-1]).reshape(1, 1)
        decoder_input_t = decoder_input_t.to(self.DEVICE)

        # Get context vector from attention network.
        context_vector, attention_weights = \
            self.model.attention(beam.decoder_states,
                                 encoder_output,
                                 x_padding_masks)

        # Replace the indexes of OOV words with the index of OOV token
        # to prevent index-out-of-bound error in the decoder.
        decoder_input_t = self.replace_oov(decoder_input_t)
        p_vocab, decoder_states = self.model.decoder(decoder_input_t,
                                                     beam.decoder_states,
                                                     encoder_output,
                                                     context_vector)

        # Calculate log probabilities.
        log_probs = torch.log(p_vocab.squeeze())
        # Filter forbidden tokens.
        if len(beam.tokens) == 1:
            forbidden_ids = [
                self.vocab[u"这"],
                self.vocab[u"此"],
                self.vocab[u"采用"],
                self.vocab[u"，"],
                self.vocab[u"。"],
                self.vocab.UNK
            ]
            log_probs[forbidden_ids] = -float('inf')

        # Get top k tokens and the corresponding logprob.
        topk_probs, topk_idx = torch.topk(log_probs, k)

        # Extend the current hypo with top k tokens, resulting k new hypos.
        best_k = [beam.extend(x,
                  log_probs[x],
                  decoder_states,
                  attention_weights,
                  beam.max_oovs,
                  beam.encoder_input) for x in topk_idx.tolist()]

        return best_k

    def beam_search(self,
                    encoder_input,
                    max_sum_len,
                    beam_width,
                    max_oovs,
                    x_padding_masks):
        """Using beam search to generate summary.

        Args:
            encoder_input (Tensor): Input sequence as the source.
            max_sum_len (int): The maximum length a summary can have.
            beam_width (int): Beam size.
            max_oovs (int): Number of out-of-vocabulary tokens.
            x_padding_masks (Tensor):
                The padding masks for the input sequences.

        Returns:
            result (list(Beam)): The list of best k candidates.
        """
        # run body_sequence input through encoder
        encoder_output, encoder_states = self.model.encoder(encoder_input)

        # initialize decoder states with encoder forward states
        decoder_states = self.model.reduce_state(encoder_states)

        # initialize the hypothesis with a class Beam instance.
        attention_weights = torch.zeros(
            (1, encoder_input.shape[1])).to(self.DEVICE)

        init_beam = Beam([self.vocab.SOS],
                         [0],
                         decoder_states,
                         attention_weights,
                         max_oovs,
                         encoder_input)

        # get the beam size and create a list for stroing current candidates
        # and a list for completed hypothesis
        k = beam_width
        curr, completed = [init_beam], []

        # use beam search for max_sum_len (maximum length) steps
        for _ in range(max_sum_len):
            # get k best hypothesis when adding a new token

            topk = []
            for beam in curr:
                # When an EOS token is generated, add the hypo to the completed
                # list and decrease beam size.
                if beam.tokens[-1] == self.vocab.EOS:
                    completed.append(beam)
                    k -= 1
                    continue
                for can in self.best_k(beam,
                                       k,
                                       encoder_output,
                                       x_padding_masks):
                    # Using topk as a heap to keep track of top k candidates.
                    # Using the sequence scores of the hypos to campare
                    # and object ids to break ties.
                    add2heap(topk, (can.seq_score(), id(can), can), k)

            curr = [items[2] for items in topk]
            # stop when there are enough completed hypothesis
            if len(completed) == k:
                break
        # When there are not engouh completed hypotheses,
        # take whatever when have in current best k as the final candidates.
        completed += curr
        # sort the hypothesis by normalized probability and choose the best one
        result = sorted(completed,
                        key=lambda x: x.seq_score(),
                        reverse=True)[0].tokens
        return result

    @timer(module='doing prediction')
    def predict(self, text, tokenize=True, beam_search=True):
        """Generate summary.

        Args:
            text (str or list): Source.
            tokenize (bool, optional):
                Whether to do tokenize or not. Defaults to True.
            beam_search (bool, optional):
                Whether to use beam search or not.
                Defaults to True (means using greedy search).

        Returns:
            str: The final summary.
        """
        if isinstance(text, str) and tokenize:
            text = list(jieba.cut(text))
        x, oov = source2ids(text, self.vocab)
        x = torch.tensor(x).to(self.DEVICE)
        max_oovs = len(oov)
        x_copy = self.replace_oov(x)
        x_copy = x_copy.unsqueeze(0)
        x_padding_masks = torch.ne(x_copy, 0).byte().float()
        if beam_search:
            summary = self.beam_search(x_copy,
                                       max_sum_len=config.max_dec_steps,
                                       beam_width=config.beam_size,
                                       max_oovs=max_oovs,
                                       x_padding_masks=x_padding_masks)
        else:
            summary = self.greedy_search(x_copy,
                                         max_sum_len=config.max_dec_steps,
                                         max_oovs=max_oovs,
                                         x_padding_masks=x_padding_masks)
        summary = outputids2words(summary,
                                  oov,
                                  self.vocab)
        return summary.replace('<SOS>', '').replace('<EOS>', '').strip()

    def replace_oov(self, input_t):
        """Replace oov tokens with <UNK> token in an input tensor.

        Args:
            input_t (Tensor): The input tensor.

        Returns:
            Tensor: All oov tokens are replaced with <UNK> token.
        """
        oov_token = torch.full(input_t.shape,
                               self.vocab.UNK).long().to(self.DEVICE)
        input_t = torch.where(input_t > len(self.vocab) - 1,
                              oov_token,
                              input_t)
        return input_t

if __name__ == "__main__":
    pred = Predict()
    print('vocab_size: ', len(pred.vocab))
    # Randomly pick a sample in test set to predict.
    with open(config.test_data_path, 'r') as test:
        picked = random.choice(list(test))
        source, ref = picked.strip().split('<sep>')

    greedy_prediction = pred.predict(source.split(),  beam_search=False)
    beam_prediction = pred.predict(source.split(),  beam_search=True)

    print('greedy: ', greedy_prediction)
    print('beam: ', beam_prediction)
    print('ref: ', ref)
