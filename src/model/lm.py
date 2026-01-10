import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional
from accelerate import Accelerator
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutput
from src.model.compressive_encoder import CompressiveEncoder


@dataclass
class CausalLMOutputForWindow(CausalLMOutput):
    window_loss: Optional[List[torch.FloatTensor]] = None
    window_valid_token_num: Optional[List[int]] = None


class LM(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        compressive_encoder: Optional[CompressiveEncoder] = None,
        window_mode: bool = False,
        lm_max_length: int = 4096,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: Optional[str] = None,
        attn_implementation: str = "flash_attention_2",
        accelerator: Optional[Accelerator] = None,
    ):
        super().__init__()
        # * init model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )

        # * set compressive encoder
        self.compressive_encoder = compressive_encoder

        # * set other parameters
        self.window_mode = window_mode
        self.lm_max_length = lm_max_length

        # * freeze model
        self.freeze_model()

        # * set accelerator
        self.accelerator = accelerator
        if device_map is None:
            if self.accelerator is not None:
                device = self.accelerator.device
            else:
                device = torch.device("cpu")
            self.model.to(device)
            if self.compressive_encoder:
                self.compressive_encoder.to(device)

    @property
    def config(self):
        return self.model.config

    @property
    def device(self):
        if self.accelerator:
            return self.accelerator.device
        else:
            return self.model.device

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        self.compressive_encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs
        )

    def save(self, output_dir):
        self.compressive_encoder.save(output_dir)

    def _two_stream_ar_forward(self, input_ids, attention_mask, labels):
        self.compressive_encoder.clear_cache()
        self.compressive_encoder.prepare(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        while not self.compressive_encoder.is_finished:
            (
                input_ids,
                attention_mask,
                labels,
                encoder_embeds,
                ph_indices,
            ) = self.compressive_encoder.step()
            inputs_embeds = self.prepare_inputs_embeds(input_ids, encoder_embeds, ph_indices)
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
            valid_token_num = (labels[:, 1:] != -100).sum()
            self.compressive_encoder.update_loss(outputs.loss, valid_token_num)

        window_loss = self.compressive_encoder.window_loss
        window_valid_token_num = self.compressive_encoder.window_valid_token_num
        sample_loss = self.compressive_encoder.sample_loss

        return CausalLMOutputForWindow(
            loss=sample_loss,
            window_loss=window_loss,
            window_valid_token_num=window_valid_token_num,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        encoder_input_ids=None,
        encoder_attention_mask=None,
        encoder_indices: Optional[List[List[int]]] = None,
        ph_indices: Optional[List[List[int]]] = None,
    ):
        if self.compressive_encoder:
            if self.window_mode:
                return self._two_stream_ar_forward(input_ids, attention_mask, labels)
            else:
                if ph_indices and encoder_indices:
                    encoder_embeds = self.get_encoder_embeds(
                        encoder_input_ids, encoder_attention_mask, encoder_indices
                    )
                    inputs_embeds = self.prepare_inputs_embeds(
                        input_ids, encoder_embeds, ph_indices
                    )
                    return self.model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                elif ph_indices is None and encoder_indices is None:
                    return self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                else:
                    raise ValueError(
                        "Arguments `ph_indices` and `encoder_indices` must be all `None` or not."
                    )
        else:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

    def get_encoder_embeds(self, encoder_input_ids, encoder_attention_mask, encoder_indices):
        encoder_embeds = []
        for idx, _encoder_indices in enumerate(encoder_indices):
            if not _encoder_indices:
                continue
            _encoder_embeds = self.compressive_encoder(
                encoder_input_ids[[idx]], encoder_attention_mask[[idx]], [_encoder_indices]
            )  # [ENCODER_LEN, H]
            encoder_embeds.append(_encoder_embeds)
        encoder_embeds = torch.cat(encoder_embeds).contiguous()  # [SUM(ENCODER_LEN), H]

        return encoder_embeds

    def prepare_inputs_embeds(self, input_ids, encoder_embeds, ph_indices: List[List[int]]):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        idx = 0
        for i, _ph_indices in enumerate(ph_indices):
            if not _ph_indices:
                continue
            inputs_embeds[i][_ph_indices] = encoder_embeds[idx : idx + len(_ph_indices)]
            idx += len(_ph_indices)

        return inputs_embeds

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        attention_mask,
        encoder_input_ids=None,
        encoder_attention_mask=None,
        encoder_indices: Optional[List[List[int]]] = None,
        ph_indices: Optional[List[List[int]]] = None,
        **gen_kwargs,
    ):
        self.eval()

        if self.compressive_encoder:
            if ph_indices and encoder_indices:
                encoder_embeds = self.get_encoder_embeds(
                    encoder_input_ids, encoder_attention_mask, encoder_indices
                )
                inputs_embeds = self.prepare_inputs_embeds(
                    input_ids, encoder_embeds, ph_indices
                )
                return self.model.generate(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )
            elif ph_indices is None and encoder_indices is None:
                return self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
            else:
                raise ValueError(
                    "Arguments `ph_indices` and `encoder_indices` must be all `None` or not."
                )
        else:
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

    def freeze_model(self):
        for _, param in self.model.named_parameters():
            param.requires_grad = False

    def _move_to_device(self, inputs):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        return inputs
