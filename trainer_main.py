import torch
import nlp
from transformers import T5Tokenizer
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1,2,6,7"
import json


import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import transformers

from transformers import T5ForConditionalGeneration, T5Tokenizer, EvalPrediction
from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorForSeq2Seq,
    T5Config,
)


logger = logging.getLogger(__name__)

def T2TDataCollator(batch: List) -> Dict[str, torch.Tensor]:
    """
    Take a list of samples from a Dataset and collate them into a batch.
    Returns:
        A dictionary of tensors
    """
    input_ids = torch.stack([example['input_ids'] for example in batch])
    lm_labels = torch.stack([example['labels'] for example in batch])
    lm_labels[lm_labels[:, :] == 0] = -100
    attention_mask = torch.stack([example['attention_mask'] for example in batch])


    return {
        'input_ids': input_ids, 
        'attention_mask': attention_mask,
        'labels': lm_labels, 
#         'decoder_attention_mask': decoder_attention_mask
    }



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file_path: Optional[str] = field(
        # default='data/codex-m/train_data.pt',
        default='data/wikidata5m/train_data.pt',
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: Optional[str] = field(
        # default='data/codex-m/valid_data.pt',
        default='data/wikidata5m/valid_data.pt',
        metadata={"help": "Path for cached valid dataset"},
    )
    max_len: Optional[int] = field(
        default=128,
        metadata={"help": "Max input length for the source text"},
    )
    target_max_len: Optional[int] = field(
        default=32,
        metadata={"help": "Max input length for the target text"},
    )




args_dict = {
  "model_name_or_path": 't5-small',
  "max_len": 128 ,
  "target_max_len": 32,
  "output_dir": './models/trainer_puri_wd5m',
  "overwrite_output_dir": True,
  "per_device_train_batch_size ": 64,
  "per_device_eval_batch_size ": 128,
  "train_batch_size": 64,
#   "gradient_accumulation_steps": 4,
  "learning_rate": 1e-4,
  "num_train_epochs": 4,
  "do_train": True
}
with open('args2.json', 'w') as f:
    json.dump(args_dict, f)


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath('args2.json'))

ta = TrainingArguments(output_dir='./models/trainer_peda_wd5m',
                      per_device_train_batch_size=80,
                    #   learning_rate=1e-4,
                      num_train_epochs=5,
                      overwrite_output_dir=True,
                      dataloader_num_workers=20,
                      prediction_loss_only=True,
                      do_train=True,
                      adafactor=True,
                      logging_steps=100,
                    #   fp16=True,
                    #   warmup_steps=2e3,
                    #   lr_scheduler_type='constant',
                    #   gradient_accumulation_steps=4
                      )


if (
    os.path.exists(ta.output_dir)
    and os.listdir(ta.output_dir)
    and ta.do_train
    and not ta.overwrite_output_dir
):
    raise ValueError(
        f"Output directory ({ta.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
    )

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if ta.local_rank in [-1, 0] else logging.WARN,
)
logger.warning(
    "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    ta.local_rank,
    ta.device,
    ta.n_gpu,
    bool(ta.local_rank != -1),
    ta.fp16,
)
logger.info("Training/evaluation parameters %s", ta)

# Set seed
set_seed(ta.seed)



config = T5Config().from_pretrained('t5-small')


model = T5ForConditionalGeneration(config)
# checkpoint_location = 'models/trainer_peda/checkpoint-5000'
# model = T5ForConditionalGeneration.from_pretrained(checkpoint_location)

print('loading data')
train_dataset  = torch.load(data_args.train_file_path)
valid_dataset = torch.load(data_args.valid_file_path)
print('loading done')

trainer = Trainer(
    model=model,
    args=ta,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=T2TDataCollator,
)

if ta.do_train:
    trainer.train(
        model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
    )
    trainer.save_model()
    # # For convenience, we also re-save the tokenizer to the same directory,
    # # so that you can share your model easily on huggingface.co/models =)
    # if trainer.is_world_master():
    #     tokenizer.save_pretrained(ta.output_dir)
