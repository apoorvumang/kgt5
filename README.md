# transformer-kgc
KG Completion on Large Graphs using Transformers (currently T5)

I don't have a requirements.txt file yet, but it needs the normal pytorch packages + huggingface transformers and huggingface accelerate

```
pip install transformers
pip install accelerate
```

There are lots of useless files in here, so please ignore them.

Dataset download: https://storage.googleapis.com/t5-kgc-colab/data/codex_training_data/codex-m/data.zip

## Usage

### Training

Set the parameter `--nproc_per_node` same as the number of GPUs that you use

```
CUDA_VISIBLE_DEVICES=1,2,3,4,5,7 python3 -m torch.distributed.launch --nproc_per_node 6 --use_env ./main_accelerate.py \
--save_prefix codex-m_6gpu \
--model_size small --dataset codex-m \
--batch_size 64 --save_steps 5000 \
--loss_steps 500
```

### Evaluation

This evaluates hits@1 unfiltered

```
CUDA_VISIBLE_DEVICES=0 python3 eval_accelerate.py --prefix codex-m_6gpu --checkpoint 90000 \
--dataset codex-m --batch_size 200
```

