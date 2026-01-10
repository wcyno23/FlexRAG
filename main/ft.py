import datasets
from dataclasses import dataclass, field
from typing import List, Optional
from transformers import HfArgumentParser, set_seed
from src.args import ModelArgs, TrainingArgs
from src.data import Data, DefaultDataCollator, FlexRAGCollator, INPUT_TAG, CONTEXT_TAG
from src.model import load_model_and_tokenizer
from src.trainer import CompressiveEncoderTrainer
from src.utils import extract_file_name_and_extension


@dataclass
class TaskArgs:
    data_files: List[str] = field(
        default_factory=lambda: [],
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
    data_splits: List[float] = field(
        default_factory=lambda: [],
    )
    down_scaling_method: Optional[str] = field(
        default="random",
    )


def down_sampling(dataset, data_splits, max_train_num_per_data, seed, idx):
    if max_train_num_per_data and len(dataset) > max_train_num_per_data:
        dataset = dataset.train_test_split(max_train_num_per_data, seed=seed)["test"]
    if len(data_splits) > 0 and data_splits[idx] < 1.0:
        dataset = dataset.train_test_split(int(len(dataset) * data_splits[idx]), seed=seed)["test"]
            
    return dataset


def prepare_ft_data(data_files, data_splits, max_train_num_per_data, seed):
    def _process(data: dict, retrieval_num: int):
        # * get prompt and replace query placehoder
        prompt = f"Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n{CONTEXT_TAG}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {INPUT_TAG}\nAnswer:"
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
                {"content": data["answers"][0], "role": "assistant"},
            ],
        }

    dataset_dict = datasets.DatasetDict()

    for idx, data_file in enumerate(data_files):
        dataset_name, _ = extract_file_name_and_extension(data_file)
        try:
            dataset = datasets.load_dataset(
                "json",
                data_files=data_file,
                split="train",
            )
            dataset = down_sampling(dataset, data_splits, max_train_num_per_data, seed, idx)
            dataset = dataset.map(
                _process,
                fn_kwargs={
                    "retrieval_num": 5,
                },
                num_proc=1,
                desc=f"prepare {dataset_name}",
                remove_columns=dataset.column_names,
            )
        except:
            dataset = datasets.load_from_disk(data_file)
            dataset = down_sampling(dataset, data_splits, max_train_num_per_data, seed, idx)
        dataset_dict[dataset_name] = dataset

    return dataset_dict


def main():
    # * set parser
    parser = HfArgumentParser([ModelArgs, TaskArgs, TrainingArgs])
    model_args, task_args, training_args = parser.parse_args_into_dataclasses()
    print(training_args)

    # * set seed
    set_seed(training_args.seed)

    # * model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args)

    # * load dataset
    with training_args.main_process_first():
        dataset_dict = prepare_ft_data(task_args.data_files, task_args.data_splits, task_args.max_train_num_per_data, training_args.seed)

        for dataset_name in dataset_dict:
            dataset = dataset_dict[dataset_name]
            dataset = dataset.map(
                Data.encode_instruction_tuning_data,
                fn_kwargs={
                    "tokenizer": tokenizer,
                    "chat_template": task_args.chat_template,
                    "lm_max_length": model_args.lm_max_length,
                    "encoder_max_length": model_args.encoder_max_length,
                    "comp_candidates": model_args.comp_candidates,
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
        print(dataset_dict)
        # * concatenate
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
    trainer.save_model()


if __name__ == "__main__":
    main()
