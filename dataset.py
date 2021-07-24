from pathlib import Path
import pkg_resources
import pickle
from collections import defaultdict
from typing import Dict, Tuple, List
import json

import numpy as np
import torch
# from qa_models import QA_model
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader
from transformers import T5TokenizerFast, GPT2Tokenizer, XLMTokenizer
import os
from typing import Dict, List
from tokenizers import Tokenizer
from unigram_tokenizer import UnigramTokenizer
from sentencepiece_tokenizer import SentencePieceTokenizer

class T5_Dataset(Dataset):
    def __init__(self, 
                split,
                dataset_name = 'wikidata5m',
                tokenizer_type = 't5',
                max_points=-1,
                relation_prediction=False,
                load_data = True, #needed because in subclass we don't want to load data
                pad_to_max = False, # max padding needed in case of TPU
                ):
        filename = 'data/{dataset_name}/{split}.txt'.format(
            dataset_name=dataset_name,
            split=split,
        )
        if relation_prediction == 1 and split == 'train':
            print('Train data contains relation prediction')
            filename = 'data/{dataset_name}/{split}.txt'.format(
                dataset_name=dataset_name,
                split='train_with_rp',
            )
        self.pad_token_id = 0
        if tokenizer_type == 't5':
            self.tokenizer = T5TokenizerFast.from_pretrained('t5-small')
        elif tokenizer_type == 'bpe':
            self.tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
            # self.tokenizer =  XLMTokenizer(vocab_file='vocab.json', merges_file='merges.txt')
        elif tokenizer_type == 'wordpiece':
            self.tokenizer = UnigramTokenizer('codexm2000')
            self.pad_token_id = self.tokenizer.pad_token_id
        elif tokenizer_type == 'wp200':
            self.tokenizer = UnigramTokenizer('codexm200')
            self.pad_token_id = self.tokenizer.pad_token_id
        elif tokenizer_type == 'wp20k':
            self.tokenizer = UnigramTokenizer('codexm20000')
            self.pad_token_id = self.tokenizer.pad_token_id
        elif tokenizer_type == 'wp_rp20k':
            self.tokenizer = UnigramTokenizer('codexm_rp_20000')
            self.pad_token_id = self.tokenizer.pad_token_id
        elif tokenizer_type == 'sentencepiece':
            self.tokenizer = SentencePieceTokenizer('wd5m_with_pad', max_tokenize_length=75, pad_to_max=pad_to_max)
            self.pad_token_id = self.tokenizer.pad_token_id
        elif tokenizer_type == 'sentencepiece-yago':
            self.tokenizer = SentencePieceTokenizer('yago_with_pad', max_tokenize_length=60, pad_to_max=pad_to_max)
            self.pad_token_id = self.tokenizer.pad_token_id
        elif tokenizer_type == 'sentencepiece-yago2':
            self.tokenizer = SentencePieceTokenizer('yago_with_pad2', max_tokenize_length=60, pad_to_max=pad_to_max)
            self.pad_token_id = self.tokenizer.pad_token_id
        elif tokenizer_type == 'metaqa_with_pad':
            self.tokenizer = SentencePieceTokenizer('metaqa_with_pad', max_tokenize_length=60, pad_to_max=pad_to_max)
            self.pad_token_id = self.tokenizer.pad_token_id
        else:
            raise NotImplementedError('{} tokenizer not implemented'.format(tokenizer_type))
        self.tokenizer_type = tokenizer_type
        self.vocab_size = self.tokenizer.vocab_size
        print('Vocab size is', self.vocab_size)
        #TODO: there is mismatch b/w tokenizer vocab size and actual vocab size in t5config
        # 32100 in tokenizer and 32128 in t5config
        # so fixing that here
        if tokenizer_type == 't5':
            self.vocab_size = 32128
        
        if load_data:
            self.data = self.loadData(filename, max_points)


        # following lines only needed by evaluate.py
        # not needed for training

        # self.splits = dict()
        # if relation_prediction == 0:
        #     self.splits["train"] = self.loadData(f"data/{dataset_name}/train.txt", max_points)
        # else:
        #     self.splits["train"] = self.loadData(f"data/{dataset_name}/train_with_rp.txt", max_points)

        # self.splits["valid"] = self.loadData(f"data/{dataset_name}/valid.txt", max_points)
        # self.splits["test"] = self.loadData(f"data/{dataset_name}/test.txt", max_points)
        # self.entity_strings = self.load_entity_strings(os.path.join("data", dataset_name, "entity_strings.txt"))
        # self.tokenized_entities = self.tokenizer(self.entity_strings, padding='max_length', truncation=True, max_length=32, return_tensors="pt")
        # self.entity_string_to_id = dict(zip(self.entity_strings, torch.arange(len(self.entity_strings)).tolist()))

    def split(self, split: str) -> Dict[List[str], List[str]]:
        return self.splits[split]

    def numLines(self, fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1
    
    def loadData(self, filename, max_points):
        file_len = self.numLines(filename)
        f = open(filename, 'r')
        inputs = []
        outputs = []
        for i in tqdm(range(file_len)):
        # for i in range(file_len):
            if i == max_points:
                break
            line = f.readline()
            if line[-1] == '\n':
                line = line[:-1]
            line = line.split('\t')
            inputs.append(line[0])
            outputs.append(line[1])
            
        data = {'inputs': inputs, 'outputs': outputs}
        return data

    @staticmethod
    def load_entity_strings(filename):
        with open(filename) as f:
            lines = f.read().splitlines()
        return lines


    def __len__(self):
        return len(self.data['inputs'])

    def __getitem__(self, index):
        data = self.data
        input = data['inputs'][index]
        output = data['outputs'][index]
        return input, output

    def _collate_fn(self, items):
        inputs = [item[0] for item in items]
        outputs = [item[1] for item in items]
        inputs_tokenized = self.tokenizer(inputs, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        outputs_tokenized = self.tokenizer(outputs, padding='max_length', truncation=True, max_length=32, return_tensors="pt")
        input_ids, attention_mask = inputs_tokenized.input_ids, inputs_tokenized.attention_mask
        labels, labels_attention_mask = outputs_tokenized.input_ids, outputs_tokenized.attention_mask
        # for labels, set -100 for padding
        labels[labels==self.pad_token_id] = -100
        # labels = -100 * torch.ones(labels.shape, dtype=torch.long)
        return input_ids, attention_mask, labels, labels_attention_mask

    def _collate_fn_new(self, items):
        inputs = [item[0] for item in items]
        outputs = [item[1] for item in items]
        inputs_tokenized = self.tokenizer(inputs, padding=True, truncation=True, max_length=128, return_tensors="pt")
        outputs_tokenized = self.tokenizer(outputs, padding=True, truncation=True, max_length=32, return_tensors="pt")
        input_ids, attention_mask = inputs_tokenized.input_ids, inputs_tokenized.attention_mask
        labels, labels_attention_mask = outputs_tokenized.input_ids, outputs_tokenized.attention_mask
        # for labels, set -100 for padding
        labels[labels==self.pad_token_id] = -100
        # labels = -100 * torch.ones(labels.shape, dtype=torch.long)
        return input_ids, attention_mask, labels, labels_attention_mask


    def _collate_eval(self, items):
        inputs = [item[0] for item in items]
        target_text = [item[1] for item in items]
        inputs_tokenized = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        return inputs_tokenized.input_ids, inputs_tokenized.attention_mask, target_text

    def _collate_eval_with_input_strings(self, items):
        inputs = [item[0] for item in items]
        target_text = [item[1] for item in items]
        inputs_tokenized = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        return inputs_tokenized.input_ids, inputs_tokenized.attention_mask, target_text, inputs


    def _collate_eval_2(self, items):
        inputs = [item[0] for item in items]
        target_text = [item[1] for item in items]
        # inputs_tokenized = self.tokenizer(inputs, return_tensors="pt")
        inputs_tokenized = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        # inputs_tokenized = self.tokenizer(inputs, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        # print(inputs_tokenized.input_ids)
        # print(inputs_tokenized.attention_mask)

        # print(inputs_tokenized.attention_mask)
        return inputs_tokenized.input_ids, inputs_tokenized.attention_mask, target_text, inputs

    def tokenizedToText(self, arr):
        return ''.join(self.tokenizer.convert_ids_to_tokens(arr))

