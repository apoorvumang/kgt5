# Complete my Sentence: Sequence to Sequence Knowledge Graph Completion and Question Answering

You can either upload the notebook `kgt5_demo_colab.ipynb` to Google Colab (https://colab.research.google.com/), or you can use the provided code. Using the notebook is recommended.

The demo notebook and this repo currently only support KGC on Wikidata5M and only hits@1 unfiltered evaluation. The full PyTorch code (which includes KGQA) will be released once the anonymity period ends.

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

