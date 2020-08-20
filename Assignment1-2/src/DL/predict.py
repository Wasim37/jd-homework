'''
@Author: xiaoyao jiang
@LastEditors: xiaoyao jiang
@Date: 2020-06-18 21:15:35
@LastEditTime: 2020-07-17 16:34:38
@FilePath: /bookClassification/src/DL/predict.py
@Desciption:
'''
from src.utils import config
import torch
from transformers import BertTokenizer
from src.DL.models.bert import Model


class Predict(object):
    def __init__(self,
                 model_path=config.root_path + '/model/saved_dict/bert.ckpt',
                 bert_path=config.root_path + '/model/bert-wwm/',
                 is_cuda=config.is_cuda):
        self.model_path = model_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.is_cuda = is_cuda
        config.bert_path = config.root_path + '/model/bert/'
        config.hidden_size = 768
        self.model = Model(config).to(config.device)
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()

    def process_data(self, text, is_cuda=config.is_cuda):
        def padding(indice, max_length, pad_idx=0):
            """
            pad 函数
            注意 token type id 右侧pad是添加1而不是0，1表示属于句子B
            """
            pad_indice = [
                item + [pad_idx] * max(0, max_length - len(item))
                for item in indice
            ]
            return torch.tensor(pad_indice)

        text_dict = self.tokenizer.encode_plus(
            text,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=config.max_length,  # Pad & truncate all sentences.
            ad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            #                                                    return_tensors='pt',     # Return pytorch tensors.
        )

        input_ids, attention_mask, token_type_ids = text_dict[
            'input_ids'], text_dict['attention_mask'], text_dict[
                'token_type_ids']

        token_ids_padded = padding([input_ids], config.max_length)
        token_type_ids_padded = padding([token_type_ids], config.max_length)
        attention_mask_padded = padding([attention_mask], config.max_length)
        return token_ids_padded, token_type_ids_padded, attention_mask_padded

    def predict(self, text):
        token_ids_padded, token_type_ids_padded, attention_mask_padded = self.process_data(
            text)
        if self.is_cuda:
            token_ids_padded = token_ids_padded.to(torch.device('cuda'))
            token_type_ids_padded = token_type_ids_padded.to(
                torch.device('cuda'))
            attention_mask_padded = attention_mask_padded.to(
                torch.device('cuda'))
        outputs = self.model(
            (token_ids_padded, attention_mask_padded, token_type_ids_padded))
        label = torch.max(outputs.data, 1)[1].cpu().numpy()[0]
        score = outputs.data[0][torch.max(
            outputs.data, 1)[1].cpu().numpy()[0]].cpu().numpy().tolist()
        return label, score