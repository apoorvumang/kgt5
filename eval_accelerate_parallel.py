from dataset import T5_Dataset
from dataset_qa import T5_DatasetQA
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
    TemperatureLogitsWarper,
)
import pickle
from trie import Trie
from accelerate import Accelerator

# with open("wd5m_entities_trie.pkl", "rb") as f:
#     entities_trie = Trie.load_from_dict(pickle.load(f))
entities_trie = None
# with open("yago2_entities_trie.pkl", "rb") as f:
#     entities_trie = Trie.load_from_dict(pickle.load(f))


def removePadding(arr):
    first_pad = (arr == 0).nonzero(as_tuple=True)[0]
    if len(first_pad) == 0:
        return arr
    else:
        last_index = first_pad[0]
        return arr[:last_index]
    
def prefixFn(batch_id, seq):
    global entities_trie
    # first token is 0 so ignoring that
    out = entities_trie.get(seq[1:].tolist())
    # if no token possible, end token
    # TODO: should be eos_token_id instead of hardcoding 2
    if out == []:
        return [2]
    else:
        return out

def grouper(arr, n):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    total = len(arr)
    if total % n != 0:
        raise ValueError('Cannot divide %d by %d' % (total, n))
    out = []
    for i in range(int(total/n)):
        start_id = i * n
        out.append(arr[start_id:start_id+n])
    return out

def getScores(ids, scores, pad_token_id):
    # ids is list of tokenized strings
    # scores is a list of tensors. each tensor contains score of each token in vocab
    # conditioned on ids till that point
    # stack scores
    scores = torch.stack(scores, dim=1)
    
    # after stacking, shape is (batch_size*num_return_sequences, num tokens in sequence, vocab size)
    # get probs
    log_probs = torch.log_softmax(scores, dim=2)
    # remove start token
    ids = ids[:,1:]
    # gather needed probs
    x = ids.unsqueeze(-1).expand(log_probs.shape)
    needed_logits = torch.gather(log_probs, 2, x)
    final_logits = needed_logits[:, :, 0]
    padded_mask = (ids == pad_token_id)
    final_logits[padded_mask] = 0
    final_scores = final_logits.sum(dim=-1)

    return final_scores.cpu().detach().numpy()


def eval(model, dataset, accelerator, args):
    num_workers = 1
    device = accelerator.device
    batch_size = args.batch_size
    # batch_size = 200
    model.cuda()
    model.eval()
    print('Using model.generate')
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=dataset._collate_eval_with_input_strings, pin_memory=True)


    model, data_loader = accelerator.prepare(model, data_loader)
    loader = tqdm(data_loader, total=len(data_loader), unit="batches")
    i = 0
    targets = []
    predictions = []
    prediction_scores = []
    model_inputs = []
    # all_entities = set(dataset.entity_strings)
    with torch.inference_mode():
        for steps, batch in enumerate(loader):
            # input_ids, attention_mask, target_text, input_text = batch
            input_ids, attention_mask, target_text, input_text = batch.inputs_tokenized.input_ids, batch.inputs_tokenized.attention_mask, batch.target_text, batch.inputs
            # TODO: for all evaluations, following was used
            # if args.task == 'kgc'
            # elif args.task == 'qa'
            if args.num_predictions > args.beam_size:
                outputs = model.module.generate(input_ids = input_ids.to(device), attention_mask=attention_mask.to(device),
                                        temperature=1.0,
                                        do_sample=True,
                                        num_return_sequences = args.num_predictions,
                                        num_beams = args.beam_size,
                                        eos_token_id = dataset.tokenizer.eos_token_id,
                                        pad_token_id = dataset.tokenizer.pad_token_id,
                                        output_scores = True,
                                        return_dict_in_generate=True,
                                        length_penalty = args.length_penalty,
                                        # top_p=0.95,
                                        # top_k=250,
                                        #  prefix_allowed_tokens_fn=prefixFn,
                                        )
            else:
                # qa only 1 output, greedy decode
                outputs = model.module.generate(input_ids = input_ids.to(device), attention_mask=attention_mask.to(device),
                                        do_sample=False,
                                        num_return_sequences = args.num_predictions,
                                        num_beams = args.beam_size,
                                        eos_token_id = dataset.tokenizer.eos_token_id,
                                        pad_token_id = dataset.tokenizer.pad_token_id,
                                        output_scores = True,
                                        return_dict_in_generate=True,
                                        length_penalty = args.length_penalty,
                                        #  prefix_allowed_tokens_fn=prefixFn,
                                        )
            # predicted_batch = outputs[:, 1:]
            # print(outputs.keys())
            # print(outputs['sequences'].shape)
            # print(outputs['sequences_scores'].shape)
            # exit(0)

            outputs = accelerator.gather(outputs)

            sequences = outputs.sequences
            
            if args.beam_size > 1:
                final_scores = outputs.sequences_scores
            else:
                scores = outputs.scores
                print(outputs)
                exit(0)
                final_scores = getScores(sequences, scores, dataset.tokenizer.pad_token_id)

            predicted_text = dataset.tokenizer.batch_decode(sequences, skip_special_tokens=True)
            # predicted_text = zip(*[iter(predicted_text)]*args.num_predictions)
            # if len(predicted_text) == current batch size, no grouping needed, otherwise group
            if len(predicted_text) == len(input_text):
                final_scores = final_scores.tolist()
            else:
                predicted_text = grouper(predicted_text, args.num_predictions) # grouping only needed if multiple predictions
                final_scores = grouper(final_scores, args.num_predictions)
                
            # for j in range(len(predicted_text)):
            #     for i in range(len(predicted_text[j])):
            #         print(predicted_text[j][i], final_scores[j][i].item())
            #     print()
            # exit(0)
            # print(len(predicted_text[0]))
            targets.extend(target_text)
            model_inputs.extend(input_text)
            predictions.extend(predicted_text)
            prediction_scores.extend(final_scores)
            
    correct = 0
    num_not_in_entities = 0
    for p, t in zip(predictions, targets):
        if args.task == 'kgc':
            if t in p:
                correct += 1
        elif args.task == 'qa':
            # TODO: in 'how much info..' it is said that exact match is done after lowercase + remove punctuation
            # trying that here
            # however, for MetaQA, we probably shouldn't do it?
            # t = [x.lower() for x in t]
            if isinstance(p, list):
                pass
                # p = [x.lower() for x in p]
            else:
                # p = [p.lower()]
                p = [p]
            if len(set(t).intersection(set(p))) > 0:
                correct += 1
            # if p in t:
            #     correct += 1
        # if p not in all_entities:
        #     num_not_in_entities += 1
    # print(num_not_in_entities/len(predictions), 'predictions were not entities')
    data_to_save = {'prediction_strings': predictions, 
                    'scores': prediction_scores,
                    'target_strings': targets,
                    'input_strings': model_inputs}
    fname = 'scores/' + args.save_file + '.pickle'
    pickle.dump(data_to_save, open(fname, 'wb'))
    accuracy = correct/len(targets)
    return accuracy    






def main():
    accelerator = Accelerator()
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix',type=str, default='temp')
    parser.add_argument('--checkpoint',type=str)
    parser.add_argument('--dataset',type=str, default='wikidata5m')
    parser.add_argument('--batch_size',type=int, default=200)
    parser.add_argument('--beam_size',type=int, default=1)
    parser.add_argument('--num_predictions',type=int, default=1)
    parser.add_argument('--length_penalty',type=float, default=1.0)
    parser.add_argument('--max_points',type=int, default=-1)
    parser.add_argument('--eval_split', type=str, default='test')
    parser.add_argument('--tokenizer', type=str, default='t5')
    parser.add_argument('--save_file', type=str, default='scores')
    parser.add_argument('--task', type=str, default='kgc')
    parser.add_argument('--hops', type=int, default=1)
                        
    args = parser.parse_args()
    print('Evaluating on split ', args.eval_split)
    if args.task == 'kgc':
        valid_dataset = T5_Dataset(args.eval_split, dataset_name=args.dataset, tokenizer_type = args.tokenizer, 
                                    max_points=args.max_points,
                                    pad_to_max=True)
    elif args.task == 'qa':
        valid_dataset = T5_DatasetQA(args.eval_split, dataset_name=args.dataset, 
                                     tokenizer_type = args.tokenizer, 
                                     max_points=args.max_points,
                                     hops=args.hops)
    checkpoint_location = 'models/{}/{}.pt'.format(args.prefix, args.checkpoint)
    print('Using %s' % checkpoint_location)
    
    model = load_accelerator_model(checkpoint_location, only_model=True)

    # if args.beam_size == 1:
    #     accuracy = eval(model, valid_dataset, args)
    # else:
    #     # accuracy = eval_multi(model, valid_dataset, args)
    #     accuracy = eval_multi_old(model, valid_dataset, args)
    accuracy = eval(model, valid_dataset, accelerator, args)
    print(accuracy)


if __name__ == "__main__":
    main()