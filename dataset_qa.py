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
from dataset import T5_Dataset
import random
from unidecode import unidecode

class T5_DatasetQA(T5_Dataset):
    def __init__(self, 
                split,
                dataset_name = 'MetaQA_half',
                tokenizer_type = 't5',
                max_points=-1,
                relation_prediction=False,
                hops=1):
        # self.super(split, dataset_name, tokenizer_type, max_points, relation_prediction)
        super().__init__(split, dataset_name, tokenizer_type, max_points, relation_prediction, load_data=False)
        filename = 'data/{dataset_name}/qa_{split}_{hops}hop.txt'.format(
            dataset_name=dataset_name,
            split=split,
            hops=hops
        )
        self.data = self.loadData(filename, max_points)
    
    def separateEntity(self, question, replacement='NE'):
        split1 = question.split('[')
        lhs = split1[0]
        split2 = split1[1].split(']')
        entity = split2[0]
        rhs = split2[1]
        final = lhs + replacement + rhs
        return final, entity

    def normalizeEntity(self, ent):
        ent = unidecode(ent)
        return ent

    def loadData(self, filename, max_points):
        print('New load data for QA')
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
            question, head_entity = self.separateEntity(line[0])
            input = 'predict answer: {0} | {1} |'.format(head_entity, question)
            output = line[1].split('|') # multiple entities can be answer
            output = [self.normalizeEntity(o) for o in output]
            inputs.append(input)
            outputs.append(output)
        data = {'inputs': inputs, 'outputs': outputs}
        return data

    def __len__(self):
        return len(self.data['inputs'])

    def __getitem__(self, index):
        data = self.data
        input = data['inputs'][index]
        output = data['outputs'][index]
        return input, output


    def _collate_fn_new(self, items):
        inputs = [item[0] for item in items]
        outputs = [random.choice(item[1]) for item in items] # randomly choose answer
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
        target_texts = [item[1] for item in items] # targets is array of entities
        inputs_tokenized = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        return inputs_tokenized.input_ids, inputs_tokenized.attention_mask, target_texts

    def _collate_eval_with_input_strings(self, items):
        inputs = [item[0] for item in items]
        target_texts = [item[1] for item in items] # targets is array of entities
        inputs_tokenized = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        return inputs_tokenized.input_ids, inputs_tokenized.attention_mask, target_texts, inputs


    def tokenizedToText(self, arr):
        return ''.join(self.tokenizer.convert_ids_to_tokens(arr))



def main():
    train_dataset = T5_DatasetQA('train', dataset_name='MetaQA_half', tokenizer_type='metaqa_with_pad', 
                            relation_prediction = False)
    i, o = train_dataset[0]
    print(i)
    print(o)


if __name__ == "__main__":
    main()
