#!/bin/bash
python --version

# Llama with retrieval
torchrun --nproc_per_node 8 -m main.eval_longbench \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--window_mode false \
--lm_max_length 3500 \
--encoder_name_or_path "" \
--encoder_num_hidden_layers 8 \
--encoder_max_length 4096 \
--attn_implementation flash_attention_2 \
--data_dir data/longbench \
--chat_template llama-2 \
--dataset_names hotpotqa 2wikimqa musique \
--comp_ratio 0 \
--batch_size 4 
