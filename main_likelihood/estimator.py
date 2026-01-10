from main_likelihood.llmlingua_compressor import MyCompressor
from typing import List
from src.utils import str_to_torch_dtype

class Estimator:
    def __init__(self, model_name_or_path, device, dtype, attn_implementation):
        self.llm_lingua = MyCompressor(
            model_name_or_path,
            device_map={"": device},
            model_config={
                "torch_dtype": str_to_torch_dtype(dtype),
                "attn_implementation": attn_implementation,
            }
        )

    def get_importance_token_indices_from_encoder_input_ids(self,
        encoder_input_ids: List[List[int]],
        tokenizer,
        text_proportion,
        use_sentence_level_filter=False,
    ) -> List[List[int]]:
        importance_token_indices = []
        for _encoder_input_ids in encoder_input_ids:
            encoder_length = len(_encoder_input_ids)
            context = tokenizer.decode(_encoder_input_ids)
            length_deviation = len(tokenizer.encode(context)) - encoder_length
            llm_lingua_output = self.llm_lingua.compress_prompt(context, target_token=text_proportion * encoder_length, use_sentence_level_filter=use_sentence_level_filter)
            token_ids_list = llm_lingua_output['token_idx']

            filtered_token_ids_list = []
            for idx in token_ids_list:
                if idx >= (length_deviation + 1):
                    filtered_token_ids_list.append(idx - length_deviation)

            importance_token_indices.append(filtered_token_ids_list)

        return importance_token_indices