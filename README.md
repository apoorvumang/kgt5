# KGT5

This is the implementation for the ACL 2022 Main Conference paper [Sequence to Sequence Knowledge Graph Completion and Question Answering](https://arxiv.org/abs/2203.10321) (KGT5).

[Click here for a demo](https://huggingface.co/spaces/apoorvumang/kgt5)

We train a sequence-to-sequence T5-small model *from scratch* - we do not initialize with the pre-trained LM weights. The task the model is trained on is head/tail prediction, where input is "\<prefix\>:\<head entity\>\<sep\>\<relation\>" and output expected is "\<tail entity\>". We use unique textual representations for each entity based on their WikiData title, and disambiguate using description/wikidata ID if necessary. For KGQA, the model pre-trained on KG link prediction is finetuned using question-answer pairs. 

<img width="878" alt="image" src="https://user-images.githubusercontent.com/1957903/160060872-60d5e5a1-f1c5-4987-804a-43375e5114e1.png">

## Resources

The main branch currently only supports KGC on Wikidata5M and only hits@1 unfiltered evaluation. Branch 'apoorv-dump' contains the latest code but it is still being cleaned. Data is yet to be uploaded. **If you need any particular data/pretrained models that we used to obtain results then please raise a github issue and we will provide it.**

For details/evaluation on WikiKG90Mv2, please see https://huggingface.co/apoorvumang/kgt5-wikikg90mv2.

To (kind of) reproduce results for WikiData5M you can use the following code.

You need pytorch packages + huggingface transformers and huggingface accelerate.

```
pip install transformers
pip install accelerate
```

Dataset download: https://storage.googleapis.com/t5-kgc-colab/data/data.zip

## Usage

### Training

#### Multi GPU
Set the parameter `--nproc_per_node` same as the number of GPUs that you use

```
CUDA_VISIBLE_DEVICES=1,2,3,4,5,7 python3 -m torch.distributed.launch --nproc_per_node 6 --use_env ./main_accelerate.py \
--save_prefix wd5m-6gpu \
--model_size small --dataset wikidata5m \
--batch_size 64 --save_steps 5000 \
--loss_steps 500
```
#### Single GPU

```
CUDA_VISIBLE_DEVICES=0 python3 main_accelerate.py \
--save_prefix wd5m-1gpu \
--model_size small --dataset wikidata5m \
--batch_size 64 --save_steps 5000 \
--loss_steps 500
```


### Evaluation

This evaluates hits@1 unfiltered

```
CUDA_VISIBLE_DEVICES=0 python3 eval_accelerate.py --prefix wd5m-6gpu --checkpoint 90000 \
--dataset wikidata5m --batch_size 200
```

