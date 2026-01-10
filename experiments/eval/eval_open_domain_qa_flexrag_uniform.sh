SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)
cd ../

torchrun --nproc_per_node 8 -m main.eval_open_domain_qa \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
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
--batch_size 8
