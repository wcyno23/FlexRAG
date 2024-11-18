import os
import datasets
import uuid
import random
import logging
from dataclasses import dataclass, field, asdict
from typing import List
from transformers import HfArgumentParser, set_seed
from transformers.tokenization_utils import PreTrainedTokenizer
from accelerate import Accelerator
from src.data import FlexRAGCollator, DefaultDataCollator
from src.open_domain_qa.config import DATASET2PROMPT, DATASET2MAXLEN, DATASET2METRIC
from src.open_domain_qa.metric import Metric
from src.chat import apply_chat_template
from src.model import load_model_and_tokenizer
from src.args import ModelArgs, LoraArgs
from torch.utils.data import DataLoader
from src.utils import save_to_json, move_to_device, FileLogger
from tqdm import tqdm
from src.data import Data, INPUT_TAG, CONTEXT_TAG
from main_likelihood.estimator import Estimator 


@dataclass
class TaskArgs:
    cpu: bool = field(
        default=False,
    )
    data_dir: str = field(
        default="data/open_domain_qa",
    )
    dataset_names: List[str] = field(
        default=lambda: ["nq", "popqa", "trivia"],
    )
    pre_data_dir: str = field(
        default="",
    )# preprocessed data dir
    chat_template: str = field(
        default="llama-2",
    )
    retrieval_num: int = field(
        default=5,
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
        default="data/results/open_domain_qa"
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

    def __post_init__(self):
        if len(self.dataset_names) == 0:
            raise ValueError("`dataset_names` can not be empty.")

def prepare_open_domain_qa(data_dir: str , dataset_names: List[str], tokenizer: PreTrainedTokenizer, retrieval_num: int, seed: int):
    def _process(data: dict, dataset_name: str, retrieval_num: int):
        # * get prompt and replace query placehoder
        prompt = DATASET2PROMPT[dataset_name]
        prompt = prompt.replace(INPUT_TAG, data["query"])

        # * format context
        retrieval_results = data["key"]
        context = "\n\n".join([
            f"Doc {i + 1}: {retrieval_results[i]}"
            for i in range(retrieval_num)
        ])

        # * get content
        content = prompt.replace(CONTEXT_TAG, context)
        
        # * format return
        return {
            "conversations": [
                {
                    "content": content,
                    "role": "user",
                    "prompt": prompt,
                    "context": context,
                },
                {"content": None, "role": "assistant"},
            ],
        }

    dataset_dict = datasets.DatasetDict()
    for dataset_name in dataset_names:
        path = os.path.join(data_dir, f"{dataset_name}.json")
        dataset = datasets.load_dataset("json", data_files=path, split="train")
        
        dataset_dict[dataset_name] = dataset.map(
            _process,
            fn_kwargs={
                "dataset_name": dataset_name,
                "retrieval_num": retrieval_num,
            },
            num_proc=32,
            desc=f"prepare {dataset_name}",
        )
    
    return dataset_dict

def load_importance_token_indices(data_dir: str, dataset_names: List[str], chat_template: str, text_proportion: float):
    dataset_dict = {}
    for dataset_name in dataset_names:
        data_name = dataset_name + "." + str(text_proportion) + ".json"
        path = os.path.join(data_dir, chat_template, data_name)
        dataset = datasets.load_dataset("json", data_files=path, split="train")
        dataset_dict[dataset_name] = dataset
    
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
    model, tokenizer = load_model_and_tokenizer(model_args, lora_args, accelerator=accelerator)
    tokenizer.padding_side = "left"

    # * load dataset and process
    with accelerator.main_process_first():
        dataset_dict = prepare_open_domain_qa(
            data_dir=task_args.data_dir,
            dataset_names=task_args.dataset_names,
            tokenizer=tokenizer,
            retrieval_num=task_args.retrieval_num,
            seed=task_args.seed,
        )
        tokenizer.add_tokens([CONTEXT_TAG], special_tokens=True)

        # * load llmlingua prepcoessed importance_token_indices_list
        importance_token_dict = load_importance_token_indices(data_dir=task_args.pre_data_dir, dataset_names=task_args.dataset_names, chat_template=task_args.chat_template,
        text_proportion=task_args.text_proportion)

        temp_dict = {}
        for dataset_name in dataset_dict.keys():
            importance_token_indices_list = importance_token_dict[dataset_name]
            temp_dict[dataset_name] = dataset_dict[dataset_name].map(
                Data.process_flexrag_likelihood_sc,
                fn_kwargs={
                    "tokenizer": tokenizer,
                    "chat_template": task_args.chat_template,
                    "lm_max_length": model_args.lm_max_length,
                    "encoder_max_length": model_args.encoder_max_length,
                    "comp_candidates": [task_args.comp_ratio],
                    "down_scaling_method": task_args.down_scaling_method,
                    "eval_mode": True,
                    "dataset_name": dataset_name, 
                    "ratio_power_of_two": task_args.ratio_power_of_two,
                    "use_encoder_at_ratio_one": task_args.use_encoder_at_ratio_one,
                    "text_proportion": task_args.text_proportion,
                    "low_comp_ratio": task_args.low_comp_ratio,
                    "importance_token_indices_list": importance_token_indices_list,
                },
                with_indices=True,
                batched=True,
                num_proc=32,
            )
             
        dataset_dict.update(temp_dict)

    # * eval open domain qa
    task_id = str(uuid.uuid4()).replace("-", "")

    metrics_dict = {}
    
    for dataset_name in task_args.dataset_names:
        if task_args.comp_ratio == 0:
            collator = DefaultDataCollator(tokenizer)
        else:
            collator = FlexRAGCollator(tokenizer)
        dataloader = DataLoader(
            dataset_dict[dataset_name],
            batch_size=task_args.batch_size,
            collate_fn=collator,
            pin_memory=True,
        )
        dataloader = accelerator.prepare(dataloader)

        # * generate
        generations = []
        for inputs in tqdm(dataloader, desc=f"eval open domain qa: {dataset_name}"):
            inputs = move_to_device(inputs, model.device)
            inputs = Data.format_inputs(inputs)
            outputs = model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=DATASET2MAXLEN[dataset_name],
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )
            outputs = outputs[:, inputs["input_ids"].shape[1]:]

            if accelerator.num_processes > 1:
                outputs = outputs.contiguous()  # must be contiguous
                outputs = accelerator.pad_across_processes(outputs, pad_index=tokenizer.pad_token_id, dim=1)
                outputs = accelerator.gather_for_metrics(outputs)
            
            outputs = outputs.tolist()
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            generations.extend(outputs)
        
        if accelerator.process_index == 0:
            # * evaluate
            answers = dataset_dict[dataset_name]["answers"]
            metrics = Metric.compute([x.split("\n")[0] for x in generations], answers, DATASET2METRIC[dataset_name])
            metrics_dict[dataset_name] = metrics
            
             # * save
            questions = dataset_dict[dataset_name]["query"]
            save_to_json(
                os.path.join(task_args.output_dir, task_id, f"{dataset_name}.json"),
                [{"question": question, "output": output, "answers": _answers} for question, output, _answers in zip(questions, generations, answers)],
            )

    # * log metric
    if accelerator.process_index == 0:
        file_logger = FileLogger(os.path.join(task_args.output_dir, "open_domain_qa.log"))
        file_logger.log(metrics_dict, ModelArgs=asdict(model_args), TaskArgs=asdict(task_args), uuid=task_id)

if __name__ == "__main__":
    main()