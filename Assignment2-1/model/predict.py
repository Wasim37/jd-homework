#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: lpx, jby
@Date: 2020-07-13 11:00:51
@LastEditTime: 2020-07-18 16:48:37
@LastEditors: Please set LastEditors
@Description: Generate a summary.
@FilePath: /JD_project_2/baseline/model/predict.py
'''


import random

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
        ###########################################
        #          TODO: module 4 task 1          #
        ###########################################

        # Get encoder output and states.
        encoder_output, encoder_states = 

        # Initialize decoder's hidden states with encoder's hidden states.
        decoder_states = 

        # Initialize decoder's input at time step 0 with the SOS token.
        decoder_input_t = 
        decoder_input_t = decoder_input_t.to(self.DEVICE, dtype=torch.int64)
        summary = 

        # Generate hypothesis with maximum decode step.
        while int(decoder_input_t.item()) != (self.vocab.EOS) \
                and len(summary) < max_sum_len:

            context_vector, attention_weights = 

            p_vocab, decoder_states = 
            # Get next token with maximum probability.
            decoder_input_t = 
            decoder_word_idx = 
            summary.append(decoder_word_idx)
            # Replace the indexes of OOV words with the index of UNK token
            # to prevent index-out-of-bound error in the decoder.
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
        ###########################################
        #          TODO: module 4 task 2.2        #
        ###########################################

        # use decoder to generate vocab distribution for the next token
        decoder_input_t = 
        decoder_input_t = decoder_input_t.to(self.DEVICE)

        # Get context vector from attention network.
        context_vector, attention_weights = 

        # Replace the indexes of OOV words with the index of UNK token
        # to prevent index-out-of-bound error in the decoder.
        decoder_input_t = self.replace_oov(decoder_input_t)
        p_vocab, decoder_states = 

        # Calculate log probabilities.
        log_probs = 

        # Get top k tokens and the corresponding logprob.
        topk_probs, topk_idx = 

        # Extend the current hypo with top k tokens, resulting k new hypos.
        best_k = 

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
        ###########################################
        #          TODO: module 4 task 2.3        #
        ###########################################

        # run body_sequence input through encoder
        encoder_output, encoder_states = 

        # initialize decoder states with encoder forward states
        decoder_states = 

        # initialize the hypothesis with a class Beam instance.
        attention_weights = 
        init_beam = Beam(

        )

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


                for can in self.best_k(beam,
                                       k,
                                       encoder_output,
                                       x_padding_masks):
                    # Using topk as a heap to keep track of top k candidates.
                    # Using the sequence scores of the hypos to campare
                    # and object ids to break ties.

            curr = 
            # stop when there are enough completed hypothesis
            if len(completed) == k:
                break
        # When there are not engouh completed hypotheses,
        # take whatever when have in current best k as the final candidates.
        completed += curr
        # sort the hypothesis by normalized probability and choose the best one
        result = 
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
        # Do tokenization if the input is raw text.
        if isinstance(text, str) and tokenize:
            text = list(jieba.cut(text))
        x, oov = source2ids(text, self.vocab)
        x = torch.tensor(x).to(self.DEVICE)
        max_oovs = len(oov)
        x_copy = self.replace_oov(x)
        x_copy = x_copy.unsqueeze(0)
        x_padding_masks = torch.ne(x_copy, 0).byte().float()
        if beam_search:  # Use beam search to decode.
            summary = self.beam_search(x_copy,
                                       max_sum_len=config.max_dec_steps,
                                       beam_width=config.beam_size,
                                       max_oovs=max_oovs,
                                       x_padding_masks=x_padding_masks)
        else:  # Use greedy search to decode.
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
