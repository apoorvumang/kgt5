import argparse
import math
from tqdm import tqdm
from dataset import T5_Dataset
from torch.utils.data import DataLoader
from utils_accelerate import *
import numpy as np


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

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        loader = tqdm(self.data_loader, total=len(self.data_loader), unit="batch")
        ranks = list()
        for steps, batch in enumerate(loader):
            ranks_in_batch = list()
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
                # add pad at beginning
                coordinates = torch.cat(
                    (
                        torch.zeros(
                            [current_chunk_size, 1],
                            dtype=torch.long,
                            device=self.device,
                        ),
                        all_entities_repeated[chunk_start:chunk_end],
                    ),
                    dim=1,
                )[:, : all_entities_repeated.shape[1]].view(current_chunk_size, -1, 1)
                # set padded logits to zero
                padded_mask = (coordinates == 0).squeeze()
                # don't set pad at beginning to zero
                padded_mask[:, 0] = False
                logits_chunk[padded_mask] = 0
                needed_logits_chunk = torch.gather(
                    logits_chunk,
                    2,
                    coordinates
                    # all_entities_repeated[chunk_start:chunk_end].view(current_chunk_size, -1, 1)
                ).view(current_chunk_size, -1)

                summed_logits = torch.sum(needed_logits_chunk, dim=1)
                summed_logit_chunks.append(summed_logits)
            summed_logits = torch.cat(summed_logit_chunks)
            for summed_logits_per_triple, label in zip(
                summed_logits.split(len(self.dataset.entity_strings)), label_strings
            ):
                arg_sorted = torch.argsort(summed_logits_per_triple, descending=True)
                rank = (
                    (arg_sorted == self.dataset.entity_string_to_id[label])
                    .nonzero(as_tuple=True)[0]
                    .item()
                )
                print(rank)
                ranks_in_batch.append(rank)
            ranks.extend(ranks_in_batch)
        ranks = np.array(ranks, dtype=np.float32)
        # add 1 to have best rank 1 not 0
        ranks += 1
        print("MR", ranks.mean())
        print("MRR", np.power(ranks, -1).mean())
        print("Hits@1", (ranks == 1).sum() / len(self.dataset))
        print("Hits@10", (ranks <= 10).sum() / len(self.dataset))
        # todo: calculate hits and so on


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

    args = parser.parse_args()
    valid_dataset = T5_Dataset(args.split, dataset_name=args.dataset)
    checkpoint_location = "models/{}/{}.pt".format(args.prefix, args.checkpoint)
    print("Using %s" % checkpoint_location)

    model = load_accelerator_model(checkpoint_location, only_model=True)
    evaluator = Evaluator(dataset=valid_dataset, model=model, args=args)
    evaluator.eval()


if __name__ == "__main__":
    main()
