SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)
cd ../

torchrun --nproc_per_node 8 -m main_likelihood.prepare_open_domain_qa \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--lm_max_length 4096 \
--encoder_name_or_path wcyno23/FlexRAG \
--encoder_num_hidden_layers 8 \
--encoder_max_length 4096 \
--attn_implementation flash_attention_2 \
--data_dir data/open_domain_qa \
--chat_template llama-2 \
--dataset_names nq popqa trivia \
--down_scaling_method stride \
--batch_size 1 \
--overall_comp_ratio 8 \
--text_proportion 0.0625 \
--retrieval_num 5 \
--use_sentence_level_filter False \
--output_dir data/likelihood_token/open_domain_qa_comp8 \

torchrun --nproc_per_node 8 -m main_likelihood.eval_open_domain_qa \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--lm_max_length 4096 \
--encoder_name_or_path wcyno23/FlexRAG \
--encoder_num_hidden_layers 8 \
--encoder_max_length 4096 \
--attn_implementation flash_attention_2 \
--data_dir data/open_domain_qa \
--chat_template llama-2 \
--dataset_names nq popqa trivia \
--down_scaling_method stride \
--batch_size 1 \
--low_comp_ratio 1 \
--overall_comp_ratio 8 \
--text_proportion 0.0625 \
--retrieval_num 5 \
--pre_data_dir data/likelihood_token/open_domain_qa_comp8 \