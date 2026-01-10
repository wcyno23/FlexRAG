import logging
from typing import Optional
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, PeftModelForCausalLM, get_peft_model
from src.args import ModelArgs, LoraArgs
from src.data import CONTEXT_TAG
from src.model.compressive_encoder import CompressiveEncoder
from src.model.lm import LM
from src.utils import str_to_torch_dtype

logger = logging.getLogger(__name__)

def load_model_and_tokenizer(
    model_args: ModelArgs,
    lora_args: LoraArgs = None,
    accelerator: Optional[Accelerator] = None,
    return_tokenizer_only: bool = False,
    down_scaling_method: str = "stride",
):
    # * First load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # * If `return_tokenizer_only` is True, we can return immediately.
    if return_tokenizer_only:
        logger.info("Only return tokenizer.")
        return tokenizer

    # * load model without compressive encoder
    if not model_args.encoder_name_or_path:
        logger.info("Load model without compressive encoder.")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=str_to_torch_dtype(model_args.dtype),
            device_map=model_args.device_map,
            attn_implementation=model_args.attn_implementation,
        )

        if lora_args and lora_args.use_lora:
            if lora_args.peft_model_name_or_path:
                model = PeftModelForCausalLM.from_pretrained(
                    model,
                    lora_args.peft_model_name_or_path,
                )
                model = model.merge_and_unload()
            else:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_args.lora_r,
                    lora_alpha=lora_args.lora_alpha,
                    target_modules=lora_args.lora_target_modules,
                    lora_dropout=lora_args.lora_dropout,
                )
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()

        if not model_args.device_map and accelerator:
            model.to(accelerator.device)
    # * load model with compressive encoder
    else:
        logger.info("Load model with compressive encoder.")
        if not model_args.window_mode:
            logger.info(
                "`window_mode` is False, so add `CONTEXT_TAG` as speical token."
            )
            tokenizer.add_tokens([CONTEXT_TAG], special_tokens=True)

        compressive_encoder = CompressiveEncoder(
            model_name_or_path=model_args.encoder_name_or_path,
            num_hidden_layers=model_args.encoder_num_hidden_layers,
            torch_dtype=str_to_torch_dtype(model_args.dtype),
            device_map=model_args.device_map,
            attn_implementation=model_args.attn_implementation,
            window=model_args.window,
            encoder_max_length=model_args.encoder_max_length,
            comp_candidates=model_args.comp_candidates,
            down_scaling_method=down_scaling_method,
        )
        model = LM(
            model_name_or_path=model_args.model_name_or_path,
            compressive_encoder=compressive_encoder,
            window_mode=model_args.window_mode,
            lm_max_length=model_args.lm_max_length,
            torch_dtype=str_to_torch_dtype(model_args.dtype),
            device_map=model_args.device_map,
            attn_implementation=model_args.attn_implementation,
            accelerator=accelerator,
        )

    return model, tokenizer