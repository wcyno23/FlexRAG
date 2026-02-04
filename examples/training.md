# Training

The training process consists of two stages:
- Pretrain
  - 90K samples from [redpajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample) with auto-regressive language modeling
  - 16K context length at maximum
  
- Finetune
  - 10K samples from [LongAlpaca](https://huggingface.co/datasets/Yukang/LongAlpaca-12k)
  - 100K samples from [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa) and [NQ](https://huggingface.co/datasets/google-research-datasets/nq_open), documents are retrieved by [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)

Please download the training data from [TacZip-Data]( https://huggingface.co/datasets/wcyno23/TacZip-Data/tree/main/train/compressive_encoder) and place it under the `data/train` directory.

## Environment Setup

```bash
conda activate flexrag
pip install transformers==4.57.3 accelerate==1.9.0 # Recommended version for stable gradient checkpointing
```

## Pretrain

```bash
# * set ddp
if [[ $WORLD_SIZE ]]; then
DDP="--nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR"
else
DDP=""
fi

# * train
OUTPUE_NAME=flexrag-llama2-pretrain
mkdir -p data/outputs/pretrain/${OUTPUE_NAME}

torchrun --nproc_per_node 8 ${DDP} -m main.pretrain \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--window_mode True \
--min_length 1024 \
--max_length 16000 \
--encoder_name_or_path meta-llama/Llama-2-7b-chat-hf \
--encoder_num_hidden_layers 8 \
--window 1024 \
--encoder_max_length 4096 \
--comp_candidates 1 2 4 8 \
--down_scaling_method random \
--data_files data/train/redpajama.json \
--output_dir data/outputs/pretrain/${OUTPUE_NAME} \
--save_strategy steps \
--save_steps 0.249999 \
--chat_template no \
--deepspeed data/ds_config/ds_config_stage2.json \
--gradient_checkpointing \
--learning_rate 5e-5 \
--attn_implementation flash_attention_2 \
--per_device_train_batch_size 1
```

## Finetune
```bash
# * set ddp
if [[ $WORLD_SIZE ]]; then
DDP="--nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR"
else
DDP=""
fi

# * train
OUTPUE_NAME=flexrag-llama2-longalpaca
mkdir -p data/outputs/ft/${OUTPUE_NAME}

BASE="data/outputs/pretrain/flexrag-llama2-pretrain"
LATEST_CKPT_DIR=$(ls -d ${BASE}/checkpoint-* | sort -V | tail -n 1)
ENCODER_PATH="${LATEST_CKPT_DIR}/compressive_encoder"
echo "Using encoder path: $ENCODER_PATH"

torchrun --nproc_per_node 8 -m main.ft \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--window_mode false \
--lm_max_length 4096 \
--encoder_name_or_path "$ENCODER_PATH" \
--encoder_num_hidden_layers 8 \
--encoder_max_length 4096 \
--comp_candidates 1 2 4 8 \
--data_files data/train/longalpaca.json \
--min_length 1024 \
--max_length 32000 \
--chat_template llama-2 \
--output_dir data/outputs/ft/${OUTPUE_NAME} \
--save_strategy epoch \
--deepspeed data/ds_config/ds_config_stage2.json \
--gradient_checkpointing \
--down_scaling_method random \
--learning_rate 1e-5 \
```

```bash
# * set ddp
if [[ $WORLD_SIZE ]]; then
DDP="--nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR"
else
DDP=""
fi

# * train
OUTPUE_NAME=flexrag-llama2-longalpaca-hotpotqa-nq
mkdir -p data/outputs/ft/${OUTPUE_NAME}

BASE="data/outputs/ft/flexrag-llama2-longalpaca"
LATEST_CKPT_DIR=$(ls -d ${BASE}/checkpoint-* | sort -V | tail -n 1)
ENCODER_PATH="${LATEST_CKPT_DIR}/compressive_encoder"
echo "Using encoder path: $ENCODER_PATH"

torchrun --nproc_per_node 8 -m main.ft \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--window_mode false \
--lm_max_length 4096 \
--encoder_name_or_path "$ENCODER_PATH" \
--encoder_num_hidden_layers 8 \
--encoder_max_length 4096 \
--comp_candidates 1 2 4 8 \
--data_files data/train/hotpotqa.json data/train/nq.json \
--min_length 1024 \
--max_length 32000 \
--learning_rate 1e-5 \
--down_scaling_method random \
--chat_template llama-2 \
--output_dir data/outputs/ft/${OUTPUE_NAME} \
--save_strategy epoch \
--max_train_num_per_data 50000 \
--deepspeed data/ds_config/ds_config_stage2.json \
--gradient_checkpointing \
```