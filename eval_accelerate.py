from dataset import T5_Dataset
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from noam_lr_scheduler import NoamLR
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import Adafactor
import transformers
import argparse
import os
from collections import OrderedDict
from utils_accelerate import *
from transformers import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    BeamSearchScorer,
)


def removePadding(arr):
    first_pad = (arr == 0).nonzero(as_tuple=True)[0]
    if len(first_pad) == 0:
        return arr
    else:
        last_index = first_pad[0]
        return arr[:last_index]
    

def eval(model, dataset, args):
    num_workers = 1
    batch_size = args.batch_size
    # batch_size = 200
    model.cuda()
    model.eval()
    print('Doing greedy decoding')
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=dataset._collate_eval)
    loader = tqdm(data_loader, total=len(data_loader), unit="batches")
    i = 0
    targets = []
    predictions = []
    for steps, batch in enumerate(loader):
        input_ids, attention_mask, target_text = batch
        outputs = model.generate(input_ids = input_ids.cuda(), attention_mask=attention_mask.cuda())
        # predicted_batch = outputs[:, 1:]
        predicted_text = dataset.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        targets.extend(target_text)
        predictions.extend(predicted_text)
            
    correct = 0
    for p, t in zip(predictions, targets):
        if p == t:
            correct += 1
    accuracy = correct/len(targets)
    return accuracy    

def getGreedyOutput(model, tokenizer, encoder_input_str):
    encoder_input_str = [encoder_input_str]
    encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
    outputs = model.generate(encoder_input_ids)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def getBeamOutput(model, tokenizer, encoder_input_str, num_beams=10, 
                  num_predictions=3, length_penalty=0.3):
    encoder_input_str = [encoder_input_str]
    encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids.cuda()
    input_ids = torch.ones((len(encoder_input_str) * num_beams, 1), device=model.device, dtype=torch.long)
    input_ids = input_ids * model.config.decoder_start_token_id
    model_kwargs = {
        "encoder_outputs": model.get_encoder()(encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True)
    }
    beam_scorer = BeamSearchScorer(
        batch_size=len(encoder_input_str),
        max_length=model.config.max_length,
        num_beams=num_beams,
        device=model.device,
        num_beam_hyps_to_keep=num_predictions,
        length_penalty=length_penalty
    )
    logits_processor = LogitsProcessorList([])
    outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def eval_multi_old(model, dataset, args):
    num_workers = 1
    batch_size = args.batch_size
    model.cuda()
    model.eval()
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=dataset._collate_eval)
    loader = tqdm(data_loader, total=len(data_loader), unit="batches")
    i = 0
    beam_size = args.beam_size
    num_predictions = args.num_predictions
    length_penalty = args.length_penalty
    correct = 0
    print('Beams: %d, Predictions: %d, Length Penalty: %f' % (beam_size, num_predictions, length_penalty))
    for steps, batch in enumerate(loader):
        encoder_input_ids, attention_mask, target_text = batch
        encoder_input_ids = encoder_input_ids.cuda()
        attention_mask = attention_mask.cuda()
        input_ids = torch.ones((len(encoder_input_ids) * beam_size, 1), device=model.device, dtype=torch.long)
        input_ids = input_ids * model.config.decoder_start_token_id
        model_kwargs = {
            "encoder_outputs": model.get_encoder()(encoder_input_ids.repeat_interleave(beam_size, dim=0), return_dict=True)
        }
        beam_scorer = BeamSearchScorer(
            batch_size=len(encoder_input_ids),
            max_length=model.config.max_length,
            num_beams=beam_size,
            device=model.device,
            num_beam_hyps_to_keep=num_predictions,
            length_penalty = length_penalty
        )
        logits_processor = LogitsProcessorList([])
        outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)
        # outputs = model.generate(input_ids = encoder_input_ids)
        # target_text = dataset.tokenizer.batch_decode(labels, skip_special_tokens=True)
        predicted_text = dataset.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        input_text = dataset.tokenizer.batch_decode(encoder_input_ids, skip_special_tokens=True)
        # print(predicted_text, input_text)
        current_batch_size = len(encoder_input_ids)
        predicted_grouped = []
        for i in range(current_batch_size):
            predicted_grouped.append(predicted_text[i*num_predictions: (i+1)*num_predictions])
        
        for i in range(current_batch_size):
            target = target_text[i]
            predicted = set(predicted_grouped[i])
            # print(predicted, target)
            if target in predicted:
                correct += 1
            
    accuracy = correct/len(dataset)
    return accuracy    



def eval_multi(model, dataset, args):
    model.eval()
    model.cuda()
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    fname = 'data/codex-m/{}.txt'.format(args.eval_split)
    f = open(fname, 'r')
    data = []
    for line in f:
        data.append(line.strip())
    f.close()

    scorer_function = getBeamOutput 
    # scorer_function = getGreedyOutput 
    if args.max_points > 0:
        num_points = args.max_points
    else:
        num_points = len(data)
    correct = 0
    for id in tqdm(range(0, num_points)):
        data_point = data[id]
        input, target = data_point.split('\t')
        predicted = set(scorer_function(model, tokenizer, input,
                                        num_beams=args.beam_size,
                                        num_predictions=args.num_predictions,
                                        length_penalty=args.length_penalty))
        print(predicted, input)
        if target in predicted:
            correct += 1
    return correct/num_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix',type=str, default='temp')
    parser.add_argument('--checkpoint',type=str)
    parser.add_argument('--dataset',type=str, default='wikidata5m')
    parser.add_argument('--batch_size',type=int, default=200)
    parser.add_argument('--beam_size',type=int, default=1)
    parser.add_argument('--num_predictions',type=int, default=1)
    parser.add_argument('--length_penalty',type=float, default=0.3)
    parser.add_argument('--max_points',type=int, default=-1)
    parser.add_argument('--eval_split', type=str, default='test')
                        
    args = parser.parse_args()
    print('Evaluating on split ', args.eval_split)
    valid_dataset = T5_Dataset(args.eval_split, dataset_name=args.dataset, max_points=args.max_points)
    checkpoint_location = 'models/{}/{}.pt'.format(args.prefix, args.checkpoint)
    print('Using %s' % checkpoint_location)
    
    model = load_accelerator_model(checkpoint_location, only_model=True)

    if args.beam_size == 1:
        accuracy = eval(model, valid_dataset, args)
    else:
        # accuracy = eval_multi(model, valid_dataset, args)
        accuracy = eval_multi_old(model, valid_dataset, args)
    
    print(accuracy)


if __name__ == "__main__":
    main()