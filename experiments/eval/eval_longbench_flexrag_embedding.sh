#!/bin/bash

python --version

torchrun --nproc_per_node 8 --master_port 29999 -m main_embedding.prepare_longbench \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--window_mode false \
--lm_max_length 3500 \
--encoder_name_or_path wcyno23/FlexRAG \
--encoder_num_hidden_layers 8 \
--encoder_max_length 4096 \
--attn_implementation flash_attention_2 \
--chat_template llama-2 \
--dataset_names hotpotqa 2wikimqa musique \
--down_scaling_method stride \
--batch_size 1 \
--low_comp_ratio 1 \
--comp_ratio 8 \
--ratio_power_of_two False \
--text_proportion  0.0625 \
--output_dir data/sentence_embedding/longbench_comp8/ \

torchrun --nproc_per_node 8 --master_port 29999 -m main_embedding.eval_longbench \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--window_mode false \
--lm_max_length 3500 \
--encoder_name_or_path wcyno23/FlexRAG \
--encoder_num_hidden_layers 8 \
--encoder_max_length 4096 \
--attn_implementation flash_attention_2 \
--chat_template llama-2 \
--dataset_names hotpotqa 2wikimqa musique \
--down_scaling_method stride \
--batch_size 4 \
--low_comp_ratio 1 \
--comp_ratio 8 \
--ratio_power_of_two False \
--text_proportion  0.0625 \
--sentence_dir data/sentence_embedding/longbench_comp8/ \