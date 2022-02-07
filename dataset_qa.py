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

class EvalBatchQA:
    def __init__(self, items, tokenizer):
        self.inputs = [item[0] for item in items]
        self.target_text = [item[1] for item in items]
        self.inputs_tokenized = tokenizer(self.inputs, padding=True, truncation=True, return_tensors="pt")

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inputs_tokenized.input_ids = self.inputs_tokenized.input_ids.pin_memory()
        self.inputs_tokenized.attention_mask = self.inputs_tokenized.attention_mask.pin_memory()
        return self


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
        max_ans_per_item = 1 # how many max answers per question, when splitting
        # for fbwq_half, we want load data same as MetaQA
        # for fbwq_full, no entity stuff, plain qa
        if dataset_name == 'fbwq_half_lego' or dataset_name == 'fbwq_half_lego_cr':
            self.data = self.loadData(filename, max_points)
            print(self.data['inputs'][0], self.data['outputs'][0])
        elif dataset_name == 'fbwq_full':
            self.data = self.loadDataSpecial(filename, max_points)
            print(self.data['inputs'][0], self.data['outputs'][0])
        elif dataset_name == 'cwq_half':
            self.data = self.loadDataNoTopicEntity(filename, max_points)
            print(self.data['inputs'][0], self.data['outputs'][0])
        else:
            max_ans_per_item = 500
            self.data = self.loadData(filename, max_points)
            print(self.data['inputs'][0], self.data['outputs'][0])
        # if train data, we want lines with too many answers to be split
        if 'train' in split:
            print('Splitting QA data for split %s' % split)
            print('Before splitting. number of points:', len(self.data['inputs']))
            self.splitPointsWithTooManyAnswers(max_ans_per_item)
            print('After splitting:', len(self.data['inputs']))

    def splitPointsWithTooManyAnswers(self, max_answers=5):
        data = self.data
        new_inputs = []
        new_outputs = []
        inputs = data['inputs']
        outputs = data['outputs']
        print('Max answers', max_answers)
        for i, o in zip(inputs, outputs):
            if len(o) <= max_answers:
                new_inputs.append(i)
                new_outputs.append(o)
            else:
                num_answers = len(o)
                for index in range(0, num_answers, max_answers):
                    answers_slice = o[index:index+max_answers]
                    new_inputs.append(i)
                    new_outputs.append(answers_slice)
        self.data['inputs'] = new_inputs
        self.data['outputs'] = new_outputs

    def getHeadFromInput(self, input):
        x = input.split(':')[1][1:]
        ent = x.split('|')[0][:-1]
        return ent
    # TODO: commented out function used in pre camera-ready results
    # def separateEntity(self, question, replacement='NE'):
    #     split1 = question.split('[')
    #     lhs = split1[0]
    #     split2 = split1[1].split(']')
    #     entity = split2[0]
    #     rhs = split2[1]
    #     final = lhs + replacement + rhs
    #     return final, entity

    def separateEntity(self, question, replacement='NE'):
        start_loc = question.find('[')
        end_loc = question.rfind(']')
        entity = question[start_loc + 1: end_loc]
        final = question[:start_loc] + replacement + question[end_loc + 1:]
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
            # TODO: earlier was normalize
            # head_entity = self.normalizeEntity(head_entity)
            # TODO: with '|' at end was the one used before camera ready
            # input = 'predict answer: {0} | {1} |'.format(head_entity, question)
            input = 'predict answer: {0} | {1}'.format(head_entity, question)
            output = line[1].split('|') # multiple entities can be answer
            # TODO: earlier was normalize
            # output = [self.normalizeEntity(o) for o in output]
            inputs.append(input)
            outputs.append(output)
        data = {'inputs': inputs, 'outputs': outputs}
        # print(data['inputs'][47])
        return data

    def loadDataSpecial(self, filename, max_points):
        print('Special load data: not formatting like head/tail prediction')
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
            # question, head_entity = self.separateEntity(line[0])
            question = line[0].replace('[', '').replace(']', '')
            input = 'predict answer: {0}'.format(question)
            output = line[1].split('|') # multiple entities can be answer
            output = [self.normalizeEntity(o) for o in output]
            inputs.append(input)
            outputs.append(output)
        data = {'inputs': inputs, 'outputs': outputs}
        return data

    def loadDataNoTopicEntity(self, filename, max_points):
        print('Special load data: no topic entity present, not formatting like head/tail prediction')
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
            # question, head_entity = self.separateEntity(line[0])
            question = line[0]
            input = 'predict answer: {0}'.format(question)
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
        return EvalBatchQA(items, self.tokenizer)
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
