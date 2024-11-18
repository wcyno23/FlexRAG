#!/bin/bash

python --version

torchrun --nproc_per_node 8 --master_port 29999 -m main.eval_open_domain_qa \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--window_mode false \
--lm_max_length 4096 \
--encoder_name_or_path wcyno23/FlexRAG \
--encoder_num_hidden_layers 8 \
--encoder_max_length 4096 \
--attn_implementation flash_attention_2 \
--data_dir data/open_domain_qa \
--chat_template llama-2 \
--dataset_names nq popqa trivia \
--retrieval_num 5 \
--comp_ratio 8 \
--down_scaling_method stride \
--batch_size 8 \
--use_encoder_at_ratio_one True 

