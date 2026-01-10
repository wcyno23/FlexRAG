SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)
cd ../

torchrun --nproc_per_node 8 -m main_likelihood.prepare_longbench \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--lm_max_length 3500 \
--encoder_name_or_path wcyno23/FlexRAG \
--encoder_num_hidden_layers 8 \
--encoder_max_length 4096 \
--attn_implementation flash_attention_2 \
--chat_template llama-2 \
--dataset_names hotpotqa 2wikimqa musique \
--down_scaling_method stride \
--batch_size 1 \
--overall_comp_ratio 8 \
--text_proportion 0.0625 \
--output_dir data/likelihood_token/longbench_comp8/ 

torchrun --nproc_per_node 8 -m main_likelihood.eval_longbench \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--lm_max_length 3500 \
--encoder_name_or_path wcyno23/FlexRAG \
--encoder_num_hidden_layers 8 \
--encoder_max_length 4096 \
--attn_implementation flash_attention_2 \
--chat_template llama-2 \
--dataset_names hotpotqa 2wikimqa musique \
--down_scaling_method stride \
--batch_size 4 \
--overall_comp_ratio 8 \
--low_comp_ratio 1 \
--text_proportion 0.0625 \
--pre_data_dir data/likelihood_token/longbench_comp8/ \
