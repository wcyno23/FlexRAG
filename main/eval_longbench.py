import os
import uuid
import logging
import datasets
from dataclasses import dataclass, field, asdict
from typing import List
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
    use_llmlingua: bool = field(
        default=False,
    )

    def __post_init__(self):
        if len(self.dataset_names) == 0:
            raise ValueError("`dataset_names` can not be empty.")

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
    model, tokenizer = load_model_and_tokenizer(model_args, lora_args, accelerator=accelerator)
    tokenizer.padding_side = "left"

    # * load dataset and process
    with accelerator.main_process_first():
        dataset_dict = prepare_longbench(
            data_dir=task_args.data_dir,
            dataset_names=task_args.dataset_names,
        )
        # RAG
        if task_args.comp_ratio == 0:
            temp_dict = {}
            for dataset_name in dataset_dict.keys():
                temp_dict[dataset_name] = dataset_dict[dataset_name].map(
                    Data.process_longbench_instruction_tuning,
                    fn_kwargs={
                        "tokenizer": tokenizer,
                        "chat_template": task_args.chat_template,
                        "lm_max_length": model_args.lm_max_length,
                        "eval_mode": True,
                        "dataset_name": dataset_name,
                    },
                    with_indices=True,
                    batched=True,
                    num_proc=32,
                )
            dataset_dict.update(temp_dict)
        # FlexRAG
        elif task_args.comp_ratio == 1:
            tokenizer.add_tokens([CONTEXT_TAG], special_tokens=True)
            dataset_dict = dataset_dict.map(
                Data.process_flexrag_instruction_tuning,
                fn_kwargs={
                    "tokenizer": tokenizer,
                    "chat_template": task_args.chat_template,
                    "lm_max_length": model_args.lm_max_length,
                    "encoder_max_length": model_args.encoder_max_length,
                    "comp_candidates": [task_args.comp_ratio],
                    "down_scaling_method": task_args.down_scaling_method,
                    "eval_mode": True,
                },
                with_indices=True,
                batched=True,
                num_proc=32,
            )
        else:
            tokenizer.add_tokens([CONTEXT_TAG], special_tokens=True)
            temp_dict = {}
            for dataset_name in dataset_dict.keys():
                temp_dict[dataset_name] = dataset_dict[dataset_name].map(
                    Data.process_flexrag_dynamic_instruction_tuning,
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
                    },
                    with_indices=True,
                    batched=True,
                    num_proc=1, # 32
                )
             
            dataset_dict.update(temp_dict)
    
    # * eval longbench
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
        for inputs in tqdm(dataloader, desc=f"eval longbench: {dataset_name}"):
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
                # FIXME: dim cannot be -1
                outputs = accelerator.pad_across_processes(outputs, pad_index=tokenizer.pad_token_id, dim=1)
                outputs = accelerator.gather_for_metrics(outputs)
            
            outputs = outputs.tolist()
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            generations.extend(outputs)
        
        if accelerator.process_index == 0:
            # * evaluate
            answers = dataset_dict[dataset_name]["answers"]
            metrics = Metric.compute([x.split("\n")[0] for x in generations], answers, DATASET2METRIC[dataset_name], all_classes=dataset_dict[dataset_name]["all_classes"])
            metrics_dict[dataset_name] = metrics
            
            # * save
            questions = dataset_dict[dataset_name]["input"]
            save_to_json(
                os.path.join(task_args.output_dir, task_id, f"{dataset_name}.json"),
                [{"question": question, "output": output, "answers": _answers} for question, output, _answers in zip(questions, generations, answers)],
            )

    # * log metric
    if accelerator.process_index == 0:
        file_logger = FileLogger(os.path.join(task_args.output_dir, "longbench.log"))
        file_logger.log(metrics_dict, ModelArgs=asdict(model_args), TaskArgs=asdict(task_args), uuid=task_id)

if __name__ == "__main__":
    main()