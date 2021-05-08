import argparse
import math
from tqdm import tqdm
from dataset import T5_Dataset
from torch.utils.data import DataLoader
from utils_accelerate import *


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
            collate_fn=dataset._collate_without_padding,
        )

    def eval(self):
        self.model.eval()
        loader = tqdm(self.data_loader, total=len(self.data_loader), unit="batch")
        for steps, batch in enumerate(loader):
            input_ids, attention_mask, labels, labels_attention_mask = batch
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)
            input_ids_repeated = torch.repeat_interleave(
                input_ids, len(self.dataset.entity_strings), dim=0
            )
            attention_mask_repeated = torch.repeat_interleave(
                attention_mask, len(self.dataset.entity_strings), dim=0
            )
            tokenized_entities = self.dataset.tokenized_entities.input_ids.to(
                self.device
            )
            all_entities_repeated = tokenized_entities.repeat([self.batch_size, 1])
            # process chunk by chunk
            for chunk_number in range(
                math.ceil(len(input_ids_repeated) / self.chunk_size)
            ):
                chunk_start = self.chunk_size * chunk_number
                chunk_end = min(
                    self.chunk_size * (chunk_number + 1), len(input_ids_repeated)
                )
                outputs_chunk = self.model(
                    input_ids=input_ids_repeated[chunk_start:chunk_end],
                    attention_mask=attention_mask_repeated[chunk_start:chunk_end],
                    labels=all_entities_repeated[chunk_start:chunk_end],
                )
                logits_chunk = outputs_chunk.logits


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

    args = parser.parse_args()
    valid_dataset = T5_Dataset("test", dataset_name=args.dataset)
    checkpoint_location = "models/{}/{}.pt".format(args.prefix, args.checkpoint)
    print("Using %s" % checkpoint_location)

    model = load_accelerator_model(checkpoint_location, only_model=True)
    evaluator = Evaluator(dataset=valid_dataset, model=model, args=args)
    evaluator.eval()


if __name__ == "__main__":
    main()
