import torch
import nlp
from transformers import T5Tokenizer
import os
# from nlp import load_dataset
from datasets import load_dataset

# import os
# os.environ['HF_DATASETS_CACHE'] = '/scratche/home/apoorv/.cache'


# HF_DATASETS_CACHE=/scratche/home/apoorv/.cache python3 make_dataset.py

tokenizer = T5Tokenizer.from_pretrained('t5-small')


train = load_dataset('text', data_files='data/wikidata5m/train.txt')['train']
valid = load_dataset('text', data_files='data/wikidata5m/valid.txt')['train']

# train = train[:5000]

def split_example(example):
    text = example['text'].split('\t')
    example['input_text'] = text[0]
    example['target_text'] = text[1]
    return example

def convert_to_features(example_batch):
    global tokenizer
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], pad_to_max_length=True, max_length=128)
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], pad_to_max_length=True, max_length=32)

    encodings = {
        'input_ids': input_encodings['input_ids'], 
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids'],
        'decoder_attention_mask': target_encodings['attention_mask']
    }

    return encodings


train = train.map(split_example, num_proc=20)
train = train.map(convert_to_features, batched=True, num_proc=20)


valid = valid.map(split_example)
valid = valid.map(convert_to_features, batched=True)

columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']
train.set_format(type='torch', columns=columns)
valid.set_format(type='torch', columns=columns)

print(len(train), len(valid))

print('Writing files')

torch.save(train, 'data/wikidata5m/train_data.pt')
torch.save(valid, 'data/wikidata5m/valid_data.pt')