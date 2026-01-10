import math
import os
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional
from transformers import AutoModel


class CompressiveEncoder(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = "meta-llama/Llama-2-7b-chat-hf",
        num_hidden_layers: int = 8,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: Optional[str] = None,
        attn_implementation: str = "flash_attention_2",
        window: int = 1024,
        encoder_max_length: int = 4096,
        comp_candidates: List[int] = [2, 4, 8, 16, 32],
        seed: int = 42,
        down_scaling_method: str = "stride",
    ):
        super().__init__()

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            # trust_remote_code=True,
            num_hidden_layers=num_hidden_layers,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            # use_safetensors=True,
        )
        self.window = window
        self.encoder_max_length = encoder_max_length
        self.comp_candidates = comp_candidates
        self.down_scaling_method = down_scaling_method
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def save(self, output_dir):
        self.model.save_pretrained(
            os.path.join(output_dir, "compressive_encoder"), safe_serialization=False
        )

    def forward(self, input_ids, attention_mask, encoder_indices: List[List[int]]):
        # * base forward
        output = self.model.forward(input_ids, attention_mask)
        output = output.last_hidden_state  # [B, S, H]
        # * select according to encoder_indices
        encoder_embeds = []
        for idx, _encoder_indices in enumerate(encoder_indices):
            encoder_embed = output[idx][_encoder_indices]  # [ENCODER_LEN, H]
            encoder_embeds.append(encoder_embed)
        encoder_embeds = torch.cat(encoder_embeds, dim=0).contiguous()  # [SUM(ENCODER_LEN), H]

        return encoder_embeds

    def clear_cache(self):
        self.idx = None
        self.input_ids_segments = None
        self.attention_mask_segments = None
        self.labels_segments = None
        self.encoder_embeds_segments = None

        self.window_loss = []
        self.window_valid_token_num = []

    def prepare(self, input_ids, attention_mask, labels):
        batch_size, seq_size = input_ids.shape
        assert batch_size == 1, "Two-Steam AR training's batch_size must be 1."

        # * prepare segment sizes
        segment_num = math.ceil(seq_size / self.window)
        comp_ratios = []
        segment_sizes = []
        comp_segment_sizes = []

        for _ in range(segment_num):
            comp_ratio = self.rng.choice(self.comp_candidates)
            segment_size = min(self.window, seq_size - sum(segment_sizes))
            comp_segment_size = (
                segment_size // comp_ratio
            )  # Since the last segment won't be compressed, we don't need to consider the case where the number is not divisible by comp_ratio

            comp_ratios.append(comp_ratio)
            segment_sizes.append(segment_size)
            comp_segment_sizes.append(comp_segment_size)

        comp_segment_sizes.pop()  # The last segment can be deleted directly (no compression needed)

        # * prepare compressive encoder segment sizes and indices
        encoder_indices = []
        _encoder_indices = []
        encoder_segement_sizes = []
        encoder_segement_size = 0

        for i in range(segment_num):
            if encoder_segement_size >= self.encoder_max_length or i == segment_num - 1:
                encoder_indices.append(_encoder_indices.copy())
                encoder_segement_sizes.append(encoder_segement_size)

                _encoder_indices.clear()
                encoder_segement_size = 0
            if i == segment_num - 1:
                break
            
            if self.down_scaling_method == "stride":
                _encoder_indices += [
                    encoder_segement_size + comp_ratios[i] * (j + 1) - 1
                    for j in range(comp_segment_sizes[i])
                ]
            elif self.down_scaling_method == "random":
                indices = torch.randperm(segment_sizes[i], device=input_ids.device)[:comp_segment_sizes[i]]
                indices = (indices + encoder_segement_size).tolist()
                _encoder_indices += indices
            else:
                raise ValueError(f"Unknown down_scaling_method: {self.down_scaling_method}")

            encoder_segement_size += segment_sizes[i]

        # * format compressive encoder inputs
        encoder_input_ids = input_ids[:, : sum(encoder_segement_sizes)].split(
            encoder_segement_sizes, dim=1
        )
        encoder_attention_mask = attention_mask[:, : sum(encoder_segement_sizes)].split(
            encoder_segement_sizes, dim=1
        )

        # * get compressive encoder embeds
        encoder_embeds = []
        for i in range(len(encoder_input_ids)):
            _encoder_embeds = self.forward(
                encoder_input_ids[i], encoder_attention_mask[i], encoder_indices[i : i + 1]
            )
            encoder_embeds.append(_encoder_embeds)
        encoder_embeds = torch.cat(encoder_embeds).contiguous()  # [SUM(ENCODER_LEN), H]

        self.idx = 1
        self.input_ids_segments = input_ids.split(segment_sizes, dim=1)
        self.attention_mask_segments = attention_mask.split(segment_sizes, dim=1)
        self.labels_segments = labels.split(segment_sizes, dim=1)
        self.encoder_embeds_segments = encoder_embeds.split(comp_segment_sizes, dim=0)

    @property
    def is_finished(self):
        return self.idx >= len(self.input_ids_segments)

    def step(self):
        # * compressive encoder embeds
        encoder_embeds = torch.cat(self.encoder_embeds_segments[: self.idx])  # [SUM(ENCODER_LEN), H]
        # * placeholder indices
        ph_indices = [[i for i in range(encoder_embeds.shape[0])]]
        # * input_ids
        input_ids = self.input_ids_segments[self.idx]
        input_ids_ph = input_ids.new_ones(1, encoder_embeds.shape[0])
        input_ids = torch.cat([input_ids_ph, input_ids], dim=1)  # [1, S]
        # * attention_mask
        attention_mask = input_ids.new_ones(input_ids.shape)  # [1, S]
        # * labels
        labels = self.labels_segments[self.idx]
        labels_ph = labels.new_full((1, encoder_embeds.shape[0]), -100)
        labels = torch.cat([labels_ph, labels], dim=1)
        # * index go forward
        self.idx += 1

        return input_ids, attention_mask, labels, encoder_embeds, ph_indices

    def update_loss(self, loss: torch.FloatTensor, valid_token_num: int):
        self.window_valid_token_num.append(valid_token_num)
        self.window_loss.append(loss)

    @property
    def sample_loss(self):
        sample_loss = 0
        sample_valid_token_num = 0
        for loss, valid_token_num in zip(self.window_loss, self.window_valid_token_num):
            if torch.isnan(loss):
                continue
            sample_loss += loss * valid_token_num
            sample_valid_token_num += valid_token_num

        return sample_loss / sample_valid_token_num
