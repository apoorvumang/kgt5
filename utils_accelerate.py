import torch
from transformers import Adafactor
import transformers
from transformers import T5Config, T5ForConditionalGeneration
from dataset import T5_Dataset


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
    try:
        args = checkpoint['args']
        print('Model args')
        print(args)
    except:
        class Args:
            model_size='small'
            optimizer='adafactor'
            learning_rate=None
            tokenizer='t5'
            dataset='wikidata5m'
        args = Args()
    if 't5' not in args.model_size:
        args.model_size = 't5-{}'.format(args.model_size)
    # if 'args.tokenizer' not in locals():
    #     args.tokenizer = 't5'
    try:
        _ = args.tokenizer
    except:
        args.tokenizer = 't5'

    #TODO: creating a temporary dataset since we need vocab size for model
    # don't need to load data
    temp_dataset = T5_Dataset('valid', dataset_name=args.dataset, tokenizer_type=args.tokenizer, load_data=False)
    kwargs = {'vocab_size': temp_dataset.vocab_size}
    config = T5Config().from_pretrained(args.model_size, **kwargs)
    
    model = T5ForConditionalGeneration(config)
    if args.optimizer == 'adafactor':
        if args.learning_rate == None:
            optimizer = Adafactor(model.parameters(), relative_step=True, warmup_init=True)
        else:
            optimizer = Adafactor(model.parameters(), lr=args.learning_rate, relative_step=False, warmup_init=False)
    elif args.optimizer == 'adam':
        optimizer = transformers.AdamW(model.parameters(), lr=args.learning_rate)
    else:
        print('Unknown optimizer type %s' % args.optimizer)
        exit(0)
    try:
        model_state_dict = checkpoint['model']
        optimizer_state_dict = checkpoint['optimizer']
        loss = checkpoint['loss']
        model_state_dict = removeModuleFromKeys(model_state_dict)
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
    except:
        model_state_dict = checkpoint
        model_state_dict = removeModuleFromKeys(model_state_dict)
        model.load_state_dict(model_state_dict)
        loss = None
    if only_model:
        return model
    else:
        return model, optimizer, args, loss

