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
import pickle
from typing import Union
from dataset import T5_Dataset, EvalBatch
from multiprocessing import shared_memory
from datasets import load_dataset
import unicodedata




class T5_Dataset_HF(T5_Dataset):
    def __init__(self, 
                split,
                dataset_name = 'web_questions',
                tokenizer_type = 't5',
                pad_to_max = False,
                max_input_sequence_length = 60,
                max_output_sequence_length = 60,
                ):
        super().__init__(split, dataset_name, tokenizer_type, max_points=-1, 
                         pad_to_max=pad_to_max,
                         max_input_sequence_length = max_input_sequence_length,
                         max_output_sequence_length = max_output_sequence_length,
                         relation_prediction=False, load_data=False)
        print('Loading huggingface dataset')
        print('Only QA task supported')
        hf_dataset = load_dataset(dataset_name)[split]
        print('Loaded dataset %s split %s of size %d' % (dataset_name, split, len(hf_dataset)))
        self.split = 'train'
        if split == 'train':
            self.split_points = True
        else:
            self.split_points = False
        self.data = self.process_dataset(hf_dataset, self.split_points)
        print('Processed dataset size: %d' % len(self.data))
        print(self[0])
        # exit(0)

    def normalize(self, s):
        s = s.replace('\t', ' ')
        s = s.replace('|', '.')
        s = unicodedata.normalize('NFKC', s)
        return s
        
    def process_dataset(self, dataset, split_points=True):
        out_data = []
        for q in dataset:
            q_text = self.normalize(q['question'])
            input = 'predict answer: ' + q_text
            if split_points == False:
                output = [self.normalize(s) for s in q['answers']]
                data_point = {'input': input, 'output': output}
                out_data.append(data_point)
            else:
                for ans in q['answers']:
                    output = self.normalize(ans)
                    data_point = {'input': input, 'output': output}
                    out_data.append(data_point)
        return out_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data
        input = data[index]['input']
        output = data[index]['output']
        return input, output
