from dataclasses import dataclass, field
from typing import List, Optional
from transformers import TrainingArguments


@dataclass
class ModelArgs:
    # * base model
    model_name_or_path: str = field(
        default="meta-llama/Llama-2-7b-chat-hf",
    )
    window_mode: bool = field(
        default=False,
    )
    lm_max_length: int = field(
        default=4096,
    )
    # * compressive encoder
    encoder_name_or_path: str = field(
        default=None,
    )
    encoder_num_hidden_layers: int = field(
        default=8,
    )
    window: int = field(
        default=1024,
    )
    encoder_max_length: int = field(
        default=4096,
    )
    comp_candidates: List[int] = field(
        default_factory=lambda: [2, 4, 8, 16, 32],
    )
    # * common
    attn_implementation: str = field(
        default="flash_attention_2",
    )
    dtype: str = field(
        default="bfloat16",
    )
    device_map: Optional[str] = field(
        default=None,
    )


@dataclass
class LoraArgs:
    use_lora: bool = field(
        default=False,
    )
    peft_model_name_or_path: str = field(
        default="",
    )
    lora_r: int = field(
        default=8,
    )
    lora_alpha: int = field(
        default=16,
    )
    lora_dropout: float = field(
        default=0.05,
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["qkv_proj", "o_proj"],
    )


@dataclass
class TrainingArgs(TrainingArguments):
    # * basic training parameters
    learning_rate: float = field(
        default=1e-4,
    )
    warmup_ratio: float = field(
        default=0.1,
    )
    num_train_epochs: float = field(
        default=1,
    )
    per_device_train_batch_size: int = field(
        default=1,
    )
    bf16: bool = field(
        default=True,
    )
    # * save and log
    output_dir: str = field(
        default="data/outputs/test",
    )
    overwrite_output_dir: bool = field(
        default=False,
    )
    # * data parameter
    dataloader_num_workers: int = field(
        default=32,
    )
    remove_unused_columns: bool = field(
        default=True,
    )
    save_strategy: str = field(
        default="epoch",
    )
    logging_steps: int = field(
        default=50,
    )
    # * avoid to use wandb
    report_to: Optional[str] = field(
        default="none",
    )
    # * other
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default_factory=lambda: {"use_reentrant": False}
    )  # If use --gradient_checkpointing, this kwargs have to be set
