import argparse
import math
from tqdm import tqdm
from dataset import T5_Dataset
from torch.utils.data import DataLoader
from utils_accelerate import *
import numpy as np
from typing import Dict
from collections import defaultdict


class Evaluator:
    def __init__(self, dataset: T5_Dataset, model, args):
        self.device = args.device
        self.dataset = dataset
        self.model = model.to(self.device)
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.chunk_size = args.chunk_size
        self.data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=dataset._collate_eval_2,
        )
        self.filter_dicts = dict()
        self.filter_dicts["train"] = self.create_filter_dict("train")
        self.filter_dicts["valid"] = self.create_filter_dict("valid")
        self.filter_dicts["test"] = self.create_filter_dict("test")

    def create_filter_dict(self, split: str) -> Dict[str, int]:
        data = self.dataset.split(split)
        filter_dict = defaultdict(list)
        for input, output in zip(data["inputs"], data["outputs"]):
            filter_dict[input].append(self.dataset.entity_string_to_id[output])
        return filter_dict

    @torch.no_grad()
    def eval(self, max_points=500):
        self.model.eval()
        loader = tqdm(self.data_loader, total=len(self.data_loader), unit="batch")
        ranks = {
            "unfiltered": list(),
            "filtered": list(),
        }
        for steps, batch in enumerate(loader):
            ranks_in_batch = {
                "unfiltered": list(),
                "filtered": list()
            }
            if steps == max_points:
                break
            input_ids, attention_mask, label_strings, input_strings = batch
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            # labels = labels.to(self.device)
            input_ids_repeated = torch.repeat_interleave(
                input_ids, len(self.dataset.entity_strings), dim=0
            )
            attention_mask_repeated = torch.repeat_interleave(
                attention_mask, len(self.dataset.entity_strings), dim=0
            )
            tokenized_entities = self.dataset.tokenized_entities.input_ids.to(
                self.device
            )
            pad_token_id = self.dataset.tokenizer.pad_token_id
            # todo: for filtering we need to use only the filtered entities per triple here
            all_entities_repeated = tokenized_entities.repeat([self.batch_size, 1])
            summed_logit_chunks = []
            # process chunk by chunk
            for chunk_number in range(
                math.ceil(len(input_ids_repeated) / self.chunk_size)
            ):
                chunk_start = self.chunk_size * chunk_number
                chunk_end = min(
                    self.chunk_size * (chunk_number + 1), len(input_ids_repeated)
                )
                current_chunk_size = chunk_end - chunk_start
                outputs_chunk = self.model(
                    input_ids=input_ids_repeated[chunk_start:chunk_end],
                    attention_mask=attention_mask_repeated[chunk_start:chunk_end],
                    labels=all_entities_repeated[chunk_start:chunk_end],
                )
                logits_chunk = outputs_chunk.logits
                soft_logits_chunk = torch.log_softmax(logits_chunk, dim=2)
                coordinates = all_entities_repeated[chunk_start:chunk_end].view(current_chunk_size, -1, 1)
                # set padded logits to zero
                # print(coordinates[0])
                padded_mask = (coordinates == pad_token_id).squeeze()
                soft_logits_chunk[padded_mask] = 0
                # print(soft_logits_chunk.shape)
                needed_soft_logits_chunk = torch.gather(
                    soft_logits_chunk,
                    2,
                    coordinates
                ).view(current_chunk_size, -1)
                # print(needed_soft_logits_chunk.shape)
                # exit(0)
                summed_logits = torch.sum(needed_soft_logits_chunk, dim=1)
                summed_logit_chunks.append(summed_logits)
            summed_logits = torch.cat(summed_logit_chunks)
            for summed_logits_per_triple, input_string, label in zip(
                summed_logits.split(len(self.dataset.entity_strings)), input_strings, label_strings
            ):
                # todo: currently we are calculating best rank on equality
                #  change to mean
                arg_sorted = torch.argsort(summed_logits_per_triple, descending=True)
                entity_id = self.dataset.entity_string_to_id[label]
                rank = (
                    (arg_sorted == entity_id)
                    .nonzero(as_tuple=True)[0]
                    .item()
                )
                print(rank)
                ranks_in_batch["unfiltered"].append(rank)

                # now filter
                true_score = summed_logits_per_triple[entity_id].clone()
                for filter_dict in self.filter_dicts.values():
                    summed_logits_per_triple[filter_dict[input_string]] = -float("inf")
                summed_logits_per_triple[entity_id] = true_score
                arg_sorted = torch.argsort(summed_logits_per_triple, descending=True)
                rank = (
                    (arg_sorted == entity_id)
                        .nonzero(as_tuple=True)[0]
                        .item()
                )
                print(rank)
                ranks_in_batch["filtered"].append(rank)
            ranks["filtered"].extend(ranks_in_batch["filtered"])
            ranks["unfiltered"].extend(ranks_in_batch["unfiltered"])
        for setting, list_of_ranks in ranks.items():
            ranks[setting] = np.array(list_of_ranks, dtype=np.float32) + 1
        # ranks = np.array(ranks, dtype=np.float32)
        # # add 1 to have best rank 1 not 0
        # ranks += 1
        total_points = len(self.dataset)
        if max_points > 0:
            total_points = max_points
        print("MR", ranks["unfiltered"].mean())
        print("MR-filtered", ranks["filtered"].mean())
        print("MRR", np.power(ranks["unfiltered"], -1).mean())
        print("MRR-filtered", np.power(ranks["filtered"], -1).mean())
        print("Hits@1", (ranks["unfiltered"] == 1).sum() / total_points)
        print("Hits@1-filtered", (ranks["filtered"] == 1).sum() / total_points)
        print("Hits@10", (ranks["unfiltered"] <= 10).sum() / total_points)
        print("Hits@10-filtered", (ranks["filtered"] <= 10).sum() / total_points)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="temp")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--dataset", type=str, default="codex-m")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_predictions", type=int, default=1)
    parser.add_argument("--length_penalty", type=float, default=0.6)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument('--tokenizer', type=str, default='t5')
    parser.add_argument('--max_points', type=int, default=500)

    args = parser.parse_args()
    valid_dataset = T5_Dataset(args.split, dataset_name=args.dataset, tokenizer_type = args.tokenizer)
    checkpoint_location = "models/{}/{}.pt".format(args.prefix, args.checkpoint)
    print("Using %s" % checkpoint_location)

    model = load_accelerator_model(checkpoint_location, only_model=True)
    evaluator = Evaluator(dataset=valid_dataset, model=model, args=args)
    evaluator.eval(max_points=args.max_points)


if __name__ == "__main__":
    main()
