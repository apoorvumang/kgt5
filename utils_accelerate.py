import torch
from transformers import Adafactor
import transformers
from transformers import T5Config, T5ForConditionalGeneration

def removeModuleFromKeys(state_dict):
    keys = list(state_dict.keys())
    if keys[0].startswith('module.'):
        new_dict = {}
        for key in state_dict.keys():
            new_key = key[7:]
            new_dict[new_key] = state_dict[key]
        return new_dict
    else:
        return state_dict

def load_accelerator_model(checkpoint_location, only_model = False):
    checkpoint = torch.load(checkpoint_location, map_location='cpu')
    args = checkpoint['args']
    config = T5Config().from_pretrained('t5-{}'.format(args.model_size))
    model = T5ForConditionalGeneration(config)
    if args.optimizer == 'adafactor':
        optimizer = Adafactor(model.parameters(), lr=args.learning_rate, relative_step=True, warmup_init=True)
    elif args.optimizer == 'adam':
        optimizer = transformers.AdamW(model.parameters(), lr=args.learning_rate)
    else:
        print('Unknown optimizer type %s' % args.optimizer)
        exit(0)
    model_state_dict = checkpoint['model']
    optimizer_state_dict = checkpoint['optimizer']
    args = checkpoint['args']
    loss = checkpoint['loss']
    model_state_dict = removeModuleFromKeys(model_state_dict)
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    if only_model:
        return model
    else:
        return model, optimizer, args, loss

