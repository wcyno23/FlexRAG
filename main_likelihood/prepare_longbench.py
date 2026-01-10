import os
import datasets
import logging
import math
from dataclasses import dataclass, field
from typing import List
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.tokenization_utils import PreTrainedTokenizer
from accelerate import Accelerator
from src.data import Data, DefaultDataCollator
from src.longbench.config import DATASET2PROMPT
from src.chat import apply_chat_template
from src.model import load_model_and_tokenizer
from src.args import ModelArgs, LoraArgs
from src.data import INPUT_TAG, CONTEXT_TAG
from main_likelihood.estimator import Estimator 


@dataclass
class TaskArgs:
    cpu: bool = field(
        default=False,
    )
    data_dir: str = field(
        default="data/longbench",
    )
    dataset_names: List[str] = field(
        default=lambda: ["hotpotqa", "2wikimqa", "musique"],
    )
    chat_template: str = field(
        default="llama-2",
    )
    max_length: int = field(
        default=3500,
    )
    overall_comp_ratio: int = field(
        default=8,
    )
    down_scaling_method: str = field(
        default="stride",
    )
    batch_size: int = field(
        default=1,
    )
    output_dir: str = field(
        default="data/results/longbench"
    )
    seed: int = field(
        default=42,
    )
    text_proportion: float = field(
        default=0.1,
    ) # the proportion of the importance context in the original context
    use_sentence_level_filter: bool = field(
        default=False,
    )

    def __post_init__(self):
        if len(self.dataset_names) == 0:
            raise ValueError("`dataset_names` can not be empty.")

# alignment between without selective compression and with selective compression
def encode_conversations_w_uniform_compression(
    inputs: dict,
    tokenizer: PreTrainedTokenizer,
    chat_template: str,
    lm_max_length: int,
    encoder_max_length: int,
    comp_ratio: int,
    text_proportion: float,
    estimator,
    use_sentence_level_filter: bool=False,
):
    conversations = inputs["conversations"]
    conversations = [
        conversations[0][0],
        {"role": "assistant", "content": None},
    ]
    # bos eos
    special_token_num = 2

    # * tokenize prompt without context, and then locate the position of the context_token_id
    prompt = conversations[0]["prompt"]
    messages = [{"role": "user", "content": prompt}] + conversations[1:]
    encoded_wo_context = apply_chat_template(
        chat_template,
        messages,
        tokenizer=tokenizer,
        return_labels=False,
    ).encoded

    # * tokenize prompt, and then split input_ids into 3 parts
    prompt_w_context = prompt.replace(CONTEXT_TAG, conversations[0]["context"])
    messages = [{"role": "user", "content": prompt_w_context}] + conversations[1:]
    encoded_w_context = apply_chat_template(
        chat_template,
        messages,
        tokenizer=tokenizer,
        return_labels=False,
    ).encoded

    # * split input_ids into 3 parts
    head_input_ids, tail_input_ids, context_input_ids = Data.split_head_tail_context(encoded_wo_context, encoded_w_context)

    # * truncate too long context
    max_encoder_token_num = (lm_max_length - len(head_input_ids) - len(tail_input_ids)) * comp_ratio
    max_encoder_token_num -= math.ceil(max_encoder_token_num / encoder_max_length) * special_token_num

    if len(context_input_ids) > max_encoder_token_num:
        half = max_encoder_token_num // 2
        context_input_ids = context_input_ids[:half] + context_input_ids[-half:]

    # * gen raw_encoder_input_ids
    if (len(context_input_ids) + len(head_input_ids) + len(tail_input_ids) - lm_max_length + 1) > 0:
        encoder_token_num = math.floor(encoder_max_length * comp_ratio * (len(context_input_ids) + len(head_input_ids) + len(tail_input_ids) - lm_max_length + 1) / (encoder_max_length * comp_ratio - encoder_max_length - special_token_num))
        half = (len(context_input_ids) - encoder_token_num) // 2
    else:
        half = 0
        
    if half > 0:
        raw_encoder_input_ids = context_input_ids[half:-half]
    else:
        raw_encoder_input_ids = context_input_ids.copy()

    # * gen encoder_input_ids
    encoder_input_ids = Data.get_encoder_input_ids(raw_encoder_input_ids, encoder_max_length, tokenizer.bos_token_id, tokenizer.eos_token_id)

    # * Use LLMLingua extracting importance token from encoder_input_ids
    importance_token_indices = estimator.get_importance_token_indices_from_encoder_input_ids(encoder_input_ids, tokenizer, text_proportion, use_sentence_level_filter)

    return importance_token_indices

def prepare_longbench(data_dir: str, dataset_names: List[str]):
    def _process(data, dataset_name):
        # * get prompt and replace query placehoder
        prompt = DATASET2PROMPT[dataset_name]
        prompt = prompt.replace(INPUT_TAG, data["input"])

        # * get content
        context = data["context"]
        content = prompt.replace(CONTEXT_TAG, context)

        # * format return
        return {"conversations": [
            {
                "content": content,
                "role": "user",
                "prompt": prompt,
                "context": context,
            },
            {"content": None, "role": "assistant"},
        ]}
    
    dataset_dict = datasets.DatasetDict()

    for dataset_name in dataset_names:
        dataset = datasets.load_from_disk(os.path.join(data_dir, dataset_name))
        dataset_dict[dataset_name] = dataset.map(
            _process,
            fn_kwargs={"dataset_name":dataset_name},
            num_proc=32,
            desc=f"prepare {dataset_name}",
        )
        
    return dataset_dict

def main():
    # * set parser
    parser = HfArgumentParser([ModelArgs, LoraArgs, TaskArgs])
    model_args, lora_args, task_args = parser.parse_args_into_dataclasses()

    # * set logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    
    # * set seed
    set_seed(task_args.seed)

    # * set device
    accelerator = Accelerator(cpu=task_args.cpu)

    # * model and tokenizer
    tokenizer = load_model_and_tokenizer(model_args, lora_args, accelerator=accelerator, return_tokenizer_only=True)
    tokenizer.padding_side = "left"

    # * load llmlingua to estimate importance
    estimator = Estimator(model_args.model_name_or_path, accelerator.device, model_args.dtype, model_args.attn_implementation)

    # * load dataset and process
    with accelerator.main_process_first():
        dataset_dict = prepare_longbench(
            data_dir=task_args.data_dir,
            dataset_names=task_args.dataset_names,
        )
    
        tokenizer.add_tokens([CONTEXT_TAG], special_tokens=True)

    for dataset_name in dataset_dict.keys():

        collator = DefaultDataCollator(tokenizer)
        dataloader = DataLoader(
            dataset_dict[dataset_name],
            batch_size=task_args.batch_size,
            collate_fn=collator,
            pin_memory=True,
        )
        dataloader = accelerator.prepare(dataloader)

        importance_token_indices_list = []
        for inputs in tqdm(dataloader, desc=f"prepare longbench likelihood: {dataset_name}"):
            importance_token_indices = encode_conversations_w_uniform_compression(
                inputs,
                tokenizer=tokenizer,
                chat_template=task_args.chat_template,
                lm_max_length=model_args.lm_max_length,
                encoder_max_length=model_args.encoder_max_length,
                comp_ratio=task_args.overall_comp_ratio,
                text_proportion=task_args.text_proportion,
                estimator=estimator,
                use_sentence_level_filter=task_args.use_sentence_level_filter,
            )

            importance_token_indices = [importance_token_indices]

            if accelerator.num_processes > 1:
                importance_token_indices = accelerator.gather_for_metrics(importance_token_indices)
            
            importance_token_indices_list.extend(importance_token_indices)

        if accelerator.process_index == 0:
            # * save
            dataset_dict[dataset_name] = dataset_dict[dataset_name].map(lambda x, idx: {'conversations': None, 'context': None, 'importance_token_indices': importance_token_indices_list[idx]}
            , with_indices=True)
            
            data_name = dataset_name + '.' + str(task_args.text_proportion) + '.json' 
            dataset_dict[dataset_name].to_json(os.path.join(task_args.output_dir, task_args.chat_template, data_name), lines=True, orient="records")

    
if __name__ == "__main__":
    main()