import torch
import datasets
from dataclasses import dataclass, field
from typing import List, Optional
from transformers import HfArgumentParser, set_seed
from src.args import ModelArgs, TrainingArgs
from src.data import Data, DefaultDataCollator, FlexRAGCollator
from src.model import load_model_and_tokenizer
from src.trainer import CompressiveEncoderTrainer
from src.utils import extract_file_name_and_extension


@dataclass
class TaskArgs:
    data_files: List[str] = field(
        default_factory=lambda: []
    )
    min_length: Optional[int] = field(
        default=None,
    )
    max_length: Optional[int] = field(
        default=None,
    )
    chat_template: Optional[str] = field(
        default=None,
    )
    max_train_num_per_data: Optional[int] = field(
        default=None,
    )
    down_scaling_method: Optional[str] = field(
        default="stride",
    )

def prepare_pretrain_data(data_files):
    dataset_dict = datasets.DatasetDict()

    for idx, data_file in enumerate(data_files):
        dataset_name, _ = extract_file_name_and_extension(data_file)
        dataset = datasets.load_dataset(
            "json",
            data_files=data_file,
            split="train",
        )
        dataset_dict[dataset_name] = dataset

    return dataset_dict


def main():
    torch.cuda.empty_cache()
    # * set parser
    parser = HfArgumentParser([ModelArgs, TaskArgs, TrainingArgs])
    model_args, task_args, training_args = parser.parse_args_into_dataclasses()
    
    # * set seed
    set_seed(training_args.seed)

    # * model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, task_args.down_scaling_method)

    # * load dataset
    with training_args.main_process_first():
        dataset_dict = prepare_pretrain_data(task_args.data_files)
        for dataset_name in dataset_dict:
            dataset = dataset_dict[dataset_name]
            dataset = dataset.map(
                Data.encode_pretraining_data,
                fn_kwargs={
                    "tokenizer": tokenizer,
                    "min_length": task_args.min_length,
                    "max_length": task_args.max_length,
                },
                batched=True,
                num_proc=32,
                remove_columns=dataset.column_names,
                batch_size=32,
                with_indices=True,
            )
            dataset_dict[dataset_name] = dataset

        dataset = datasets.concatenate_datasets(dataset_dict.values())

    # * set trainer
    if model_args.window_mode:
        collator = DefaultDataCollator(tokenizer)
    else:
        collator = FlexRAGCollator(tokenizer)
    trainer = CompressiveEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    model.accelerator = trainer.accelerator

    # * training
    trainer.train()

if __name__ == "__main__":
    main()