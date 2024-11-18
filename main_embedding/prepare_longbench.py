import os
import datasets
import uuid
import logging
import random
import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from transformers import HfArgumentParser, set_seed
from accelerate import Accelerator
from src.data import FlexRAGCollator, DefaultDataCollator
from src.longbench.config import DATASET2PROMPT, DATASET2TASK, DATASET2MAXLEN, DATASET2METRIC
from src.longbench.metric import Metric
from src.chat import apply_chat_template
from src.model import load_model_and_tokenizer
from src.args import ModelArgs, LoraArgs
from torch.utils.data import DataLoader
from src.utils import save_to_json, move_to_device, FileLogger
from tqdm import tqdm
from src.data import Data, INPUT_TAG, CONTEXT_TAG
from transformers.tokenization_utils import PreTrainedTokenizer
from main_embedding.embedder import get_sentence_begin_indices, get_sentence_priority_list, merge
from main_embedding.embedder import SentenceEmbedder


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
    comp_ratio: int = field(
        default=16,
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
    ratio_power_of_two: bool = field( 
        default=True,
    ) # whether use ratio which is power of two for dynamic compression
    use_encoder_at_ratio_one: bool = field(
        default=False,
    )
    text_proportion: float = field(
        default=0.1,
    ) # the proportion of the importance context in the original context  
    low_comp_ratio: int = field(
        default=1,
    ) # the compression ratio for important context
    use_sentence_level_filter: bool = field(
        default=False,
    )

    def __post_init__(self):
        if len(self.dataset_names) == 0:
            raise ValueError("`dataset_names` can not be empty.")

# alignment between without selective compression and with selective compression
def process_flexrag_dynamic_instruction_tuning(
    inputs: dict,
    tokenizer: PreTrainedTokenizer,
    chat_template: str,
    lm_max_length: int,
    encoder_max_length: int,
    comp_candidates: List[int],
    dataset_name: str,
    text_proportion: float,
    sentence_embedder, 
    low_comp_ratio: int=1,
    down_scaling_method: str="random",
    min_length: Optional[int]=None,
    max_length: Optional[int]=None,
    eval_mode: bool=False,
    ratio_power_of_two: bool=True,
    use_encoder_at_ratio_one: bool=False,
):
    if not eval_mode:
        outputs["labels"] = []
    conversations = inputs["conversations"]
    # * if eval, reformat conversations
    if eval_mode:
        conversations = [
            conversations[0][0],
            {"role": "assistant", "content": None},
        ]
        if dataset_name in ["hotpotqa", "2wikimqa", "musique"]:
            special_token_num = 2
        else:
            special_token_num = 1

        # * tokenize prompt without context, and then locate the position of the context_token_id
        prompt = conversations[0]["prompt"]
        messages = [{"role": "user", "content": prompt}] + conversations[1:]
        encoded_wo_context = apply_chat_template(
            chat_template,
            messages,
            tokenizer=tokenizer,
            return_labels=not eval_mode,
        ).encoded

        length_wo_context = len(encoded_wo_context["input_ids"])

        # * tokenize prompt, and then split input_ids into 3 parts
        prompt_w_context = prompt.replace(CONTEXT_TAG, conversations[0]["context"])
        messages = [{"role": "user", "content": prompt_w_context}] + conversations[1:]  # fmt: skip
        encoded_w_context = apply_chat_template(
            chat_template,
            messages,
            tokenizer=tokenizer,
            return_labels=not eval_mode,
        ).encoded
        length_w_context = len(encoded_w_context["input_ids"])

        # * split input_ids into 3 parts
        head_input_ids = []
        for j in range(length_wo_context):
            if encoded_wo_context["input_ids"][j] != encoded_w_context["input_ids"][j]:
                break
            head_input_ids.append(encoded_w_context["input_ids"][j])
        tail_input_ids = []

        for j in range(1, length_wo_context + 1):
            if encoded_wo_context["input_ids"][-j] != encoded_w_context["input_ids"][-j]:
                break
            tail_input_ids.append(encoded_w_context["input_ids"][-j])
        tail_input_ids = tail_input_ids[::-1]
        context_input_ids = encoded_w_context["input_ids"][len(head_input_ids):-len(tail_input_ids)]

        comp_ratio = random.choice(comp_candidates)
       
        # * truncate too long context
        max_encoder_token_num = (lm_max_length - len(head_input_ids) - len(tail_input_ids)) * comp_ratio
        max_encoder_token_num -= math.ceil(max_encoder_token_num / encoder_max_length) * special_token_num

        if (len(context_input_ids) > max_encoder_token_num) and not use_encoder_at_ratio_one:
            half = max_encoder_token_num // 2
            context_input_ids = context_input_ids[:half] + context_input_ids[-half:]

        # * gen raw_encoder_input_ids
        if (len(context_input_ids) + len(head_input_ids) + len(tail_input_ids) - lm_max_length + 1) > 0:
            encoder_token_num = math.floor(encoder_max_length * comp_ratio * (len(context_input_ids) + len(head_input_ids) + len(tail_input_ids) - lm_max_length + 1) / (encoder_max_length * comp_ratio - encoder_max_length - special_token_num))

            half = (len(context_input_ids) - encoder_token_num) // 2
        else:
            half = 0
        if half > 0:
            normal_token_input_ids_pair = (
                context_input_ids[:half],
                context_input_ids[-half:],
            )
            raw_encoder_input_ids = context_input_ids[half:-half]
        else:
            normal_token_input_ids_pair = ([], [])
            raw_encoder_input_ids = context_input_ids.copy()

        # * gen encoder_input_ids
        encoder_input_ids = []
        for j in range(math.ceil(len(raw_encoder_input_ids) / (encoder_max_length - special_token_num))):
            start = j * (encoder_max_length - special_token_num)
            end = min(start + encoder_max_length - special_token_num, len(raw_encoder_input_ids))
            if special_token_num == 1:
                encoder_input_ids.append([tokenizer.bos_token_id] + raw_encoder_input_ids[start:end])
            else:
                encoder_input_ids.append([tokenizer.bos_token_id] + raw_encoder_input_ids[start:end] + [tokenizer.eos_token_id])

        # * Use embedding model to extract importance token from encoder_input_ids
        number_of_sentences, sentences_list, sentence_begin_indices = get_sentence_begin_indices(encoder_input_ids, tokenizer, sentence_embedder.sentence_extractor)
        sentence_priority_list, sentences_ids_list = get_sentence_priority_list(sentences_list, sentence_embedder, prompt, number_of_sentences)
        
    return sentence_begin_indices, sentence_priority_list, sentences_ids_list

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

    # * load semtemce_embedder
    sentence_embedder = SentenceEmbedder(accelerator=accelerator)

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

        sentence_begin_indices_list = []
        sentence_priority_lists = []
        sentences_ids_lists = []
        for inputs in tqdm(dataloader, desc=f"prepare longbench sentence embedding: {dataset_name}"):
            sentence_begin_indices, sentence_priority_list, sentences_ids_list = process_flexrag_dynamic_instruction_tuning(inputs, tokenizer=tokenizer,
                chat_template=task_args.chat_template,
                lm_max_length=model_args.lm_max_length,
                encoder_max_length=model_args.encoder_max_length,
                comp_candidates=[task_args.comp_ratio],
                down_scaling_method=task_args.down_scaling_method,
                eval_mode=True,
                dataset_name=dataset_name, 
                ratio_power_of_two=task_args.ratio_power_of_two,
                use_encoder_at_ratio_one=task_args.use_encoder_at_ratio_one,
                text_proportion=task_args.text_proportion,
                low_comp_ratio=task_args.low_comp_ratio,
                sentence_embedder=sentence_embedder,
                )

            sentence_begin_indices = [sentence_begin_indices]
            sentence_priority_list = [sentence_priority_list]
            sentences_ids_list = [sentences_ids_list]

            if accelerator.num_processes > 1:
                sentence_begin_indices = accelerator.gather_for_metrics(sentence_begin_indices)
                sentence_priority_list = accelerator.gather_for_metrics(sentence_priority_list)
                sentences_ids_list = accelerator.gather_for_metrics(sentences_ids_list)
            
            sentence_begin_indices_list.extend(sentence_begin_indices)
            sentence_priority_lists.extend(sentence_priority_list)
            sentences_ids_lists.extend(sentences_ids_list)

        if accelerator.process_index == 0:
            # * save
            dataset_dict[dataset_name] = dataset_dict[dataset_name].map(lambda x, idx: {'conversations': None, 'context': None,'sentence_begin_indices': sentence_begin_indices_list[idx], 'sentence_priority_list': sentence_priority_lists[idx], 'sentences_ids_list': sentences_ids_lists[idx]}
            , with_indices=True)
            
            data_name = dataset_name + '.' + str(task_args.text_proportion) + '.json' 
            dataset_dict[dataset_name].to_json(os.path.join(task_args.output_dir, task_args.chat_template, data_name), lines=True, orient="records")

    

if __name__ == "__main__":
    main()