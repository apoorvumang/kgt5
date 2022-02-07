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


# https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
# see what he says about stringarray
class StringArray:
    def __init__(self, strings, encoding = 'utf_32_le'):
        strings = list(strings)
        self.encoding = encoding
        self.multiplier = dict(ascii = 1, utf_16_le = 2, utf_32_le = 4)[encoding]
        self.data = torch.ByteTensor(torch.ByteStorage.from_buffer(''.join(strings).encode(encoding)))
        self.cumlen = torch.LongTensor(list(map(len, strings))).cumsum(dim = 0).mul_(self.multiplier)
        assert int(self.cumlen[-1]) == len(self.data), f'[{encoding}] is not enough to hold characters, use a larger character class'

    def __getitem__(self, i):
        return bytes(self.data[(self.cumlen[i - 1] if i >= 1 else 0) : self.cumlen[i]]).decode(self.encoding)

    def __len__(self):
        return len(self.cumlen)
    
    def tolist(self):
        data_bytes, cumlen = bytes(self.data), self.cumlen.tolist()
        return [data_bytes[0:cumlen[0]].decode(self.encoding)] + [data_bytes[start:end].decode(self.encoding) for start, end in zip(cumlen[:-1], cumlen_mul[1:])]




class T5_Dataset_Large(T5_Dataset):
    def __init__(self, 
                split,
                dataset_name = 'wikikg90mv2',
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
        print('Loading large dataset (integer format pickle dump)')
        ent_alias_file = 'data/{dataset_name}/ent_alias_list.pickle'.format(dataset_name=dataset_name)
        rel_alias_file = 'data/{dataset_name}/rel_alias_list.pickle'.format(dataset_name=dataset_name)
        
        print('Loading aliases')

        # see issue https://github.com/pytorch/pytorch/issues/13246
        # and https://github.com/pytorch/pytorch/issues/20433
        # without making np.array a memory leak was happening
        # user mem usage: ps --no-headers -eo user,rss | awk '{arr[$1]+=$2}; END {for (i in arr) {print i,arr[i]}}' | sort -nk2
        # with num_workers=0 works really slow
        # shared_memory.ShareableList causes crash TypeError: function takes exactly 5 arguments (1 given)
        self.ent2alias = StringArray(pickle.load(open(ent_alias_file, 'rb')))
        self.rel2alias = StringArray(pickle.load(open(rel_alias_file, 'rb')))
        print('Loading data')
        self.data_int = np.array(self.load_from_pickle(dataset_name, split))
        

    def load_from_pickle(self, dataset_name, split):
        filename = 'data/{dataset_name}/{split}_int.pickle'.format(
            dataset_name=dataset_name,
            split=split,
        )
        return pickle.load(open(filename, 'rb'))


    def __len__(self):
        return len(self.data_int)

    
    def __getitem__(self, index):
        t = self.data_int[index]
        input = self.ent2alias[t[0]] + '| ' + self.rel2alias[t[1]]
        output = self.ent2alias[t[2]]
        return input, output