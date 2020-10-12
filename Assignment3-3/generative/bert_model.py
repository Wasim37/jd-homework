#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Peixin Lin
Date: 2020-09-29 17:05:15
LastEditTime: 2020-09-30 09:58:48
FilePath: /Assignment3-3/generative/bert_model.py
Desciption: Implement the BERT model.
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''
import math

import torch
from torch import nn


def swish(x):
    return x * torch.sigmoid(x)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {
    "gelu": gelu,
    "relu": torch.nn.functional.relu,
    "swish": swish,
    "mish": mish
}


class BertConfig(object):
    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
    ):

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


class BertLayerNorm(nn.Module):
    """LayerNorm层, 见Transformer(一), 讲编码器(encoder)的第3部分"""
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style
        (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()


    def forward(self, x):

        return


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super().__init__()


    def forward(self, input_ids=None, token_type_ids=None, position_ids=None):

        return


class BertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of "
                "attention heads (%d)" %
                (config.hidden_size, config.num_attention_heads))
        

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)

        # 最后xshape (batch_size, num_attention_heads, seq_len, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, output_attentions=False):


        # 得到输出
        if output_attentions:
            return context_layer, attention_probs
        return context_layer, None


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        

    def forward(self, hidden_states, input_tensor):
        
        return


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        

    def forward(self, hidden_states, attention_mask, output_attentions=False):
        
        return attention_output, attention_metrix


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
       
    def forward(self, hidden_states):
        

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        
    def forward(self, hidden_states, input_tensor):
        
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
    def forward(self, hidden_states, attention_mask, output_attentions=False):
        
        return layer_output, attention_matrix


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        

    def forward(self,
                hidden_states,
                attention_mask,
                output_all_encoded_layers=True,
                output_attentions=False):
       
        return all_encoder_layers, all_attention_matrices


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        

    def forward(self, hidden_states):
        
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
       

    def forward(self, hidden_states):
        
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        

    def forward(self, sequence_output):

        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()


    def forward(self, pooled_output):

        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        

    def forward(self, sequence_output, pooled_output):
        
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear)):
            # 初始线性映射层的参数为正态分布
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            # 初始化LayerNorm中的alpha为全1, beta为全0
            module.beta.data.zero_()
            module.gamma.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            # 初始化偏置为0
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.
    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762
    """
    def __init__(self, config):
        super().__init__(config)
        

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                output_all_encoded_layers=True,
                output_attentions=False):

        
        return encoder_layers, pooled_output
