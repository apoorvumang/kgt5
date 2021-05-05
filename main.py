from dataset import T5_Dataset
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from noam_lr_scheduler import NoamLR
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import Adafactor
import transformers

def removePadding(arr):
    first_pad = (arr == 0).nonzero(as_tuple=True)[0]
    if len(first_pad) == 0:
        return arr
    else:
        last_index = first_pad[0]
        return arr[:last_index]
    

def eval(model, dataset, args=None):
    num_workers = 1
    batch_size = 200
    model.cuda()
    model.eval()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=dataset._collate_without_padding)
    loader = tqdm(data_loader, total=len(data_loader), unit="batches")
    i = 0
    targets = []
    predictions = []
    for steps, batch in enumerate(loader):
        input_ids, attention_mask, labels, labels_attention_mask = batch
        outputs = model.generate(input_ids = input_ids.cuda())
        actual_batch = labels
        predicted_batch = outputs[:, 1:]
        for i in range(len(actual_batch)):
            predict = removePadding(predicted_batch[i])
            actual = removePadding(actual_batch[i])
            predictions.append(predict.cpu().numpy())
            targets.append(actual.cpu().numpy())
            
    correct = 0
    for p, t in zip(predictions, targets):
        p_text = dataset.tokenizedToText(p)
        t_text = dataset.tokenizedToText(t)
        if p_text == t_text:
            correct += 1
    accuracy = correct/len(targets)
    return accuracy    



def train(model, dataset, valid_dataset, args=None):
    num_workers = 30
    batch_size = 80
    loss_steps = 100
    save_steps = 5000
    use_scheduler = True
    if use_scheduler:
        # optimizer = torch.optim.Adam(model.parameters())
        optimizer = Adafactor(model.parameters(), relative_step=True, warmup_init=True)
    else:
        optimizer = Adafactor(
            model.parameters(),
            lr=0.0001,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )

    # if use_scheduler:
    #     scheduler = NoamLR(optimizer)
        # scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, 
        #     num_warmup_steps = 10000, num_training_steps = 50000)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            collate_fn=dataset._collate_fn)
    
    model.cuda()
    model.train()
    num_steps = 0
    for epoch in range(5):
        loader = tqdm(data_loader, total=len(data_loader), unit="batches")
        running_loss = 0
        for steps, batch in enumerate(loader):
            num_steps += 1
            input_ids, attention_mask, labels, labels_attention_mask = batch
            optimizer.zero_grad()
            # print(input_ids)
            # print(labels)
            # exit(0)
            outputs = model(input_ids = input_ids.cuda(), 
            attention_mask = attention_mask.cuda(), 
            labels= labels.cuda()
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            # if use_scheduler:
            #     scheduler.step()
            if num_steps % save_steps == 0:
                print('Validating at step %d' % num_steps)
                accuracy = eval(model, valid_dataset)
                print('Accuracy: ', accuracy)
                print('Saving at step %d' % num_steps)
                folder_name = 'models/wikidata_{}.pt'.format(num_steps)
                model.save_pretrained(folder_name)
            if num_steps % loss_steps == 0:
                print('Loss: ', running_loss/loss_steps)
                running_loss = 0
            running_loss += loss.item()
        print('epoch loss ', running_loss)


# config = T5Config()
# config.decoder_start_token_id = 0

config = T5Config().from_pretrained('t5-small')

# print(config)

# train_dataset = T5_Dataset('train', dataset_name='codex-m')
# valid_dataset = T5_Dataset('valid', dataset_name='codex-m')

train_dataset = T5_Dataset('train', dataset_name='wikidata5m')
valid_dataset = T5_Dataset('valid', dataset_name='wikidata5m')


model = T5ForConditionalGeneration(config)
# model = T5ForConditionalGeneration.from_pretrained('t5-small')
# checkpoint_iter = 35000
# model = T5ForConditionalGeneration.from_pretrained('models/codex_m_{}.pt'.format(checkpoint_iter))

train(model, train_dataset, valid_dataset)




# accuracy = eval(model, dataset)
# print(accuracy)