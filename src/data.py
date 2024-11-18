import math
import random
import datasets
import torch
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset, Sampler
from transformers import set_seed
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging
from src.chat import apply_chat_template
from main_embedding.embedder import get_sentence_begin_indices, get_sentence_priority_list, merge

logger = logging.get_logger(__name__)


PH_TOKEN_ID = 100

INPUT_TAG = "[INPUT_RmehNsY1]"
CONTEXT_TAG = "[CONTEXT_RmehNsY1]"


# * Main Data Class

class Data:
    @staticmethod
    def format_inputs(inputs: dict) -> dict:
        keys_to_reserve = ["input_ids", "attention_mask", "encoder_input_ids", "encoder_attention_mask", "ph_indices", "encoder_indices", "labels"]
        new_inputs = {}
        for k in inputs:
            if k in keys_to_reserve:
                new_inputs[k] = inputs[k]
        return new_inputs
    
    @staticmethod
    def split_input_ids(input_ids_a, input_ids_b):
        min_length = min(len(input_ids_a), len(input_ids_b))

        head_input_ids = []

    @staticmethod
    def get_encoder_input_ids(
        context_input_ids: List[int], encoder_max_length: int, bos_token_id: Optional[int] = None
    ) -> List[List[int]]:
        encoder_input_ids = []
        step = encoder_max_length - 1
        for i in range(0, len(context_input_ids), step):
            if bos_token_id:
                encoder_input_ids.append(
                    [bos_token_id] + context_input_ids[i:i+step]
                )
            else:
                encoder_input_ids.append(
                    context_input_ids[i:i+step]
                )
        return encoder_input_ids

    @staticmethod
    def get_encoder_indices(
        encoder_input_ids: List[List[int]], comp_ratio: int, method="stride", token_idx_list=None
    ) -> List[List[int]]:
        assert method in ["stride", "random", "terminal"], "Down scalng method is error. Make sure method in `['stride', 'random', 'terminal']`."
        
        encoder_indices = []
        for idx, _encoder_input_ids in enumerate(encoder_input_ids):
            _encoder_indices = list(range(comp_ratio - 1, len(_encoder_input_ids), comp_ratio))
            if len(_encoder_input_ids) % comp_ratio != 0:
                _encoder_indices.append(len(_encoder_input_ids) - 1)
            if comp_ratio == 1:
                _encoder_indices = _encoder_indices[1:]
            # * insert important token idx
            if token_idx_list is not None:
                _encoder_indices_set = set(_encoder_indices)
                for token_id in token_idx_list[idx]:
                    _encoder_indices_set.add(token_id)
                _encoder_indices = list(_encoder_indices_set)
            encoder_indices.append(_encoder_indices)

        if method == "stride":
            pass
        elif method == "random":
            new_encoder_indices = []
            for i in range(len(encoder_indices)):
                num = len(encoder_indices[i])
                _encoder_indices = random.sample(range(len(encoder_input_ids[i])), num)
                _encoder_indices = sorted(_encoder_indices)
                new_encoder_indices.append(_encoder_indices)
            encoder_indices = new_encoder_indices
        elif method == "terminal":
            new_encoder_indices = []
            for i in range(len(encoder_indices)):
                num = len(encoder_indices[i])
                _encoder_indices = list(range(len(encoder_input_ids[i])))[-num:]
                new_encoder_indices.append(_encoder_indices)
            encoder_indices = new_encoder_indices

        return encoder_indices


    @staticmethod
    def get_token_level_weighted_encoder_indices(
        encoder_input_ids: List[List[int]], high_comp_ratio: int, selected_token_indices: List[List[int]], method="stride",
    ) -> List[List[int]]:
        assert method in ["stride", "random", "terminal"], "Down scalng method is error. Make sure method in `['stride', 'random', 'terminal']`."
        encoder_indices = []

        # * insert selected_token_indices into uniform indices
        for idx, _encoder_input_ids in enumerate(encoder_input_ids):
            _encoder_indices = list(range(high_comp_ratio - 1, len(_encoder_input_ids), high_comp_ratio))
            if len(_encoder_input_ids) % high_comp_ratio != 0:
                _encoder_indices.append(len(_encoder_input_ids) - 1)
            if high_comp_ratio == 1:
                _encoder_indices = _encoder_indices[1:]
            # * insert selected token indices
            if selected_token_indices is not None:
                _encoder_indices_set = set(_encoder_indices)
                for token_index in selected_token_indices[idx]:
                    _encoder_indices_set.add(token_index)
                _encoder_indices = list(_encoder_indices_set)
            encoder_indices.append(_encoder_indices)

        return encoder_indices


    @staticmethod
    def get_sentence_level_weighted_encoder_indices(
        encoder_input_ids: List[List[int]], high_comp_ratio: int, low_comp_ratio: int, encoder_max_length: int, selected_sentences_id: List[int], sentence_begin_indices, sentences_ids_list: List[List[int]], method="stride", 
    ) -> List[List[int]]:
        assert method in ["stride", "random", "terminal"], "Down scalng method is error. Make sure method in `['stride', 'random', 'terminal']`."
        
        encoder_indices = []
        encoder_index = []
        step = encoder_max_length - 1

        low_ratio_length = 0
        high_ratio_length = 0

        for sentences_ids in sentences_ids_list:
            _encoder_indices = []
            for sentence_id in sentences_ids:
                if sentence_id in selected_sentences_id:
                    sentence_length = sentence_begin_indices[sentence_id + 1]['idx2'] - sentence_begin_indices[sentence_id]['idx2']
                    __encoder_indices = list(range(low_comp_ratio - 1 + sentence_begin_indices[sentence_id]['idx2'], sentence_begin_indices[sentence_id + 1]['idx2'], low_comp_ratio))
                    if sentence_length % low_comp_ratio != 0:
                        __encoder_indices.append(sentence_begin_indices[sentence_id + 1]['idx2'] - 1)
                    if low_comp_ratio == 1:
                        __encoder_indices = __encoder_indices[1:]
                    _encoder_indices.extend(__encoder_indices)

                    low_ratio_length += len(__encoder_indices)
                else:
                    sentence_length = sentence_begin_indices[sentence_id + 1]['idx2'] - sentence_begin_indices[sentence_id]['idx2']
                    __encoder_indices = list(range(high_comp_ratio - 1 + sentence_begin_indices[sentence_id]['idx2'], sentence_begin_indices[sentence_id + 1]['idx2'], high_comp_ratio))
                    if sentence_length % high_comp_ratio != 0:
                        __encoder_indices.append(sentence_begin_indices[sentence_id + 1]['idx2'] - 1)
                    if high_comp_ratio == 1:
                        __encoder_indices = __encoder_indices[1:]
                    _encoder_indices.extend(__encoder_indices)

                    high_ratio_length += len(__encoder_indices)
            encoder_indices.append(_encoder_indices)

        return encoder_indices

 
    @staticmethod
    def process_instruction_tuning(
        data, indices, tokenizer, chat_template, lm_max_length: Optional[int] = None, min_length:Optional[int]=None, max_length:Optional[int]=None, eval_mode:bool=False
    ):
        outputs = {
            "input_ids": [],
            "attention_mask": [],
            "length": [],
            "index": [],
        }
        if not eval_mode:
            outputs["labels"] = []

        for i, conversations in enumerate(data["conversations"]):
            # * if eval, reformat conversations
            if eval_mode:
                pass
                # * pretraining
                conversations = [
                    conversations[0],
                    {"role": "assistant", "content": None},
                ]
                # * instruction tuning
                # conversations = conversations
            encoded = apply_chat_template(
                chat_template,
                conversations,
                tokenizer=tokenizer,
                # only return labels in evaluation mode
                return_labels=not eval_mode,
            ).encoded

            # skip data that not fall in between min_length and max_length
            min_length = min_length if min_length else float("-inf")
            max_length = max_length if max_length else float("inf")
            if (
                len(encoded["input_ids"]) <= min_length
                or len(encoded["input_ids"]) > max_length
            ):
                continue
            for k, v in encoded.items():
                if k in outputs:
                    if lm_max_length and len(v) > lm_max_length:
                        v = v[:lm_max_length // 2] + v[-lm_max_length // 2:]
                    outputs[k].append(v)
            outputs["length"].append(len(encoded["input_ids"]))
            outputs["index"].append(indices[i])

        return outputs

    @staticmethod
    def process_longbench_instruction_tuning(
        data, indices, tokenizer, chat_template, dataset_name, lm_max_length: Optional[int] = None, min_length:Optional[int]=None, max_length:Optional[int]=None, eval_mode:bool=False
    ):
        outputs = {
            "input_ids": [],
            "attention_mask": [],
            "length": [],
            "index": [],
        }
        if not eval_mode:
            outputs["labels"] = []

        for i, conversations in enumerate(data["conversations"]):

            # * if eval, reformat conversations
            if eval_mode:
                conversations = [
                    conversations[0],
                    {"role": "assistant", "content": None},
                ]

            encoded = apply_chat_template(
                chat_template,
                conversations,
                tokenizer=tokenizer,
                # only return labels in evaluation mode
                return_labels=not eval_mode,
            ).encoded

            # skip data that not fall in between min_length and max_length
            min_length = min_length if min_length else float("-inf")
            max_length = max_length if max_length else float("inf")
            if (
                len(encoded["input_ids"]) < min_length
                or len(encoded["input_ids"]) > max_length
            ):
                continue

            for k, v in encoded.items():
                if k in outputs:
                    if lm_max_length and len(v) > lm_max_length:
                        v = v[:lm_max_length // 2] + v[-lm_max_length // 2:]
                            
                    outputs[k].append(v)
            outputs["length"].append(len(encoded["input_ids"]))
            outputs["index"].append(indices[i])

        return outputs

    @staticmethod
    def process_flexrag_instruction_tuning(
        data: List[dict],
        indices: List[int],
        tokenizer: PreTrainedTokenizer,
        chat_template: str,
        lm_max_length: int,
        encoder_max_length: int,
        comp_candidates: List[int],
        down_scaling_method: str="random",
        min_length: Optional[int]=None,
        max_length: Optional[int]=None,
        eval_mode: bool=False,
    ):
        outputs = {
            "input_ids": [],
            "encoder_input_ids": [],
            "ph_indices": [],
            "encoder_indices": [],
            "length": [],
            "index": [],
        }
        if not eval_mode:
            outputs["labels"] = []

        for i, conversations in enumerate(data["conversations"]):
            # * if eval, reformat conversations
            if eval_mode:
                conversations = [
                    conversations[0],
                    {"role": "assistant", "content": None},
                ]

            # * select compression ratio
            comp_ratio = random.choice(comp_candidates)  

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

            # * filter data that not fall in between min_length and max_length
            min_length = min_length if min_length else float("-inf")
            max_length = max_length if max_length else float("inf")
            if length_w_context < min_length or length_w_context > max_length:
                continue

            # * truncate context_input_ids
            context_length = len(context_input_ids)
            remain_length = (
                lm_max_length - (length_w_context - context_length)
            ) * comp_ratio
            if remain_length < context_length:
                half = remain_length // 2
                context_input_ids = context_input_ids[:half] + context_input_ids[-half:]

            # * encoder_input_ids
            encoder_input_ids = Data.get_encoder_input_ids(context_input_ids, encoder_max_length, tokenizer.bos_token_id)
            # * encoder_indices
            encoder_indices = Data.get_encoder_indices(encoder_input_ids, comp_ratio, down_scaling_method)
            # * input_ids and ph_indices
            ph_indices_num = sum([len(x) for x in encoder_indices])
            ph_indices = [len(head_input_ids) + j for j in range(ph_indices_num)]
            input_ids = head_input_ids + [PH_TOKEN_ID] * ph_indices_num + tail_input_ids

            # * format
            outputs["input_ids"].append(input_ids)
            outputs["encoder_input_ids"].append(encoder_input_ids)
            outputs["ph_indices"].append(ph_indices)
            outputs["encoder_indices"].append(encoder_indices)

            # * if not eval, process labels
            if not eval_mode:
                head_labels = encoded_w_context["labels"][:len(head_input_ids)]
                tail_labels = encoded_w_context["labels"][-len(tail_input_ids):]
                labels = head_labels + [-100] * ph_indices_num + tail_labels
                outputs["labels"].append(labels)

            outputs["length"].append(len(input_ids))
            outputs["index"].append(indices[i])

        return outputs

    # @staticmethod
    def process_flexrag_dynamic_instruction_tuning(
        data: List[dict],
        indices: List[int],
        tokenizer: PreTrainedTokenizer,
        chat_template: str,
        lm_max_length: int,
        encoder_max_length: int,
        comp_candidates: List[int],
        dataset_name: str,
        down_scaling_method: str="random",
        min_length: Optional[int]=None,
        max_length: Optional[int]=None,
        eval_mode: bool=False,
        ratio_power_of_two: bool=True,
        use_encoder_at_ratio_one: bool=False,
        profile=False,
    ):
        outputs = {
            "input_ids": [],
            "encoder_input_ids": [],
            "ph_indices": [],
            "encoder_indices": [],
            "length": [],
            "index": [],
        }
        if not eval_mode:
            outputs["labels"] = []

        for i, conversations in enumerate(data["conversations"]):
            # * if eval, reformat conversations
            if eval_mode:
                conversations = [
                    conversations[0],
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

            if comp_ratio == 1 and not use_encoder_at_ratio_one:
                input_ids = encoded_w_context["input_ids"]
                ph_indices = []
                encoder_input_ids = []
                encoder_indices = []
            elif comp_ratio == 1:
                # * filter data that not fall in between min_length and max_length
                min_length = min_length if min_length else float("-inf")
                max_length = max_length if max_length else float("inf")
                if length_w_context < min_length or length_w_context > max_length:
                    continue

                # * truncate context_input_ids
                context_length = len(context_input_ids)
                remain_length = (
                    lm_max_length - (length_w_context - context_length)
                ) * comp_ratio
                if remain_length < context_length:
                    half = remain_length // 2
                    context_input_ids = context_input_ids[:half] + context_input_ids[-half:]
        
                # * encoder_input_ids
                encoder_input_ids = Data.get_encoder_input_ids(context_input_ids, encoder_max_length, tokenizer.bos_token_id)
                # * encoder_indices
                encoder_indices = Data.get_encoder_indices(encoder_input_ids, comp_ratio, down_scaling_method)
                # * input_ids and ph_indices
                ph_indices_num = sum([len(x) for x in encoder_indices])
                ph_indices = [len(head_input_ids) + j for j in range(ph_indices_num)]
                input_ids = head_input_ids + [PH_TOKEN_ID] * ph_indices_num + tail_input_ids
            else:
                # * gen raw_encoder_input_ids
                if (len(context_input_ids) + len(head_input_ids) + len(tail_input_ids) - lm_max_length + 1) > 0:
                    encoder_token_num = math.floor(encoder_max_length * comp_ratio * (len(context_input_ids) + len(head_input_ids) + len(tail_input_ids) - lm_max_length + 1) / (encoder_max_length * comp_ratio - encoder_max_length - special_token_num))
                    half = (len(context_input_ids) - encoder_token_num) // 2
                else:
                    half = 0
                # for open domain qa profile test
                if lm_max_length == 3500 and profile:
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
                    
                # * encoder_indices
                encoder_indices = Data.get_encoder_indices(encoder_input_ids, comp_ratio, down_scaling_method)
                len_for_encoder = lm_max_length - len(head_input_ids) - len(tail_input_ids) - len(normal_token_input_ids_pair[0]) - len(normal_token_input_ids_pair[1])

                # * input_ids
                ph_indices_num = sum([len(x) for x in encoder_indices])
                input_ids = head_input_ids + normal_token_input_ids_pair[0] + [PH_TOKEN_ID] * ph_indices_num + normal_token_input_ids_pair[1] + tail_input_ids
                # * gen placeholder_indices
                ph_indices = [k + len(head_input_ids) + len(normal_token_input_ids_pair[0]) for k in range(ph_indices_num)]
            
            # * format
            outputs["input_ids"].append(input_ids)
            outputs["encoder_input_ids"].append(encoder_input_ids)
            outputs["ph_indices"].append(ph_indices)
            outputs["encoder_indices"].append(encoder_indices)

            # * if not eval, process labels
            if not eval_mode:
                head_labels = encoded_w_context["labels"][:len(head_input_ids)]
                tail_labels = encoded_w_context["labels"][-len(tail_input_ids):]
                labels = head_labels + [-100] * ph_indices_num + tail_labels

                outputs["labels"].append(labels)

            outputs["length"].append(len(input_ids))
            outputs["index"].append(indices[i])

        return outputs

    def process_flexrag_likelihood_sc(
        data: List[dict],
        indices: List[int],
        tokenizer: PreTrainedTokenizer,
        chat_template: str,
        lm_max_length: int,
        encoder_max_length: int,
        comp_candidates: List[int],
        dataset_name: str,
        text_proportion: float,
        importance_token_indices_list: List[List[List[int]]], 
        low_comp_ratio: int=1,
        down_scaling_method: str="random",
        min_length: Optional[int]=None,
        max_length: Optional[int]=None,
        eval_mode: bool=False,
        ratio_power_of_two: bool=True,
        use_encoder_at_ratio_one: bool=False,
    ):
        outputs = {
            "input_ids": [],
            "encoder_input_ids": [],
            "ph_indices": [],
            "encoder_indices": [],
            "length": [],
            "index": [],
        }
        if not eval_mode:
            outputs["labels"] = []
        
        for i, conversations in enumerate(data["conversations"]):
            # * if eval, reformat conversations
            if eval_mode:
                conversations = [
                    conversations[0],
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
             # * cal compression_ratio
            if comp_ratio <= 0:
                remain_length = lm_max_length - len(head_input_ids) - len(tail_input_ids)
                if ratio_power_of_two:
                    comp_ratio = 2 ** math.ceil(math.log2(math.ceil(len(context_input_ids) / remain_length))) 
                else:
                    comp_ratio = math.ceil(len(context_input_ids) / remain_length)
            else:
                # * select compression ratio
                comp_ratio = comp_ratio

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

            # * Load previous extracted importance_token_from_LLMLingua
            importance_token_indices = importance_token_indices_list[indices[i]]['importance_token_indices']

            # * Select token_ids based on low_comp_ratio
            len_encoder_input_ids = 0
            for _encoder_input_ids in encoder_input_ids:
                len_encoder_input_ids += len(_encoder_input_ids)

            low_ratio_index_length = 0 
            selected_token_indices = []

            for _importance_token_indices in importance_token_indices:
                temp = []
                for idx, token_index in enumerate(_importance_token_indices):
                    if idx % low_comp_ratio == 0:
                        temp.append(token_index)
                if len(_importance_token_indices) % low_comp_ratio != 0:
                    temp.append(_importance_token_indices[-1])
                low_ratio_index_length += len(temp)
                selected_token_indices.append(temp)

            # * Based on the length of imporatance sentences after compress, calculate proper ratio for remain context
            if (len(context_input_ids) + len(head_input_ids) + len(tail_input_ids) - lm_max_length + 1) < 0:
                # open domain qa
                len_for_encoder = math.ceil(len(raw_encoder_input_ids) / comp_ratio)
                remain_length = len_for_encoder - low_ratio_index_length 
            else:
                # longbench    
                len_for_encoder = lm_max_length - len(head_input_ids) - len(tail_input_ids) - len(normal_token_input_ids_pair[0]) - len(normal_token_input_ids_pair[1])
                remain_length = len_for_encoder - low_ratio_index_length - len(encoder_input_ids)

            if remain_length <= 0:
                high_comp_ratio = 0
            else:
                if ratio_power_of_two:
                    high_comp_ratio = 2 ** math.ceil(math.log2(math.ceil((len_encoder_input_ids - low_ratio_index_length) / remain_length)))
                else:
                    high_comp_ratio = math.ceil((len_encoder_input_ids - low_ratio_index_length) / remain_length)

            # * encoder_indices
            if low_ratio_index_length > 0 and high_comp_ratio > 0 and high_comp_ratio < 40:
                encoder_indices = Data.get_token_level_weighted_encoder_indices(encoder_input_ids, high_comp_ratio, selected_token_indices, down_scaling_method)

                # make sure that lm_length is less than len_for_encoder
                lm_length = sum([len(x) for x in encoder_indices])
                while lm_length > len_for_encoder:
                    high_comp_ratio += 1
                    encoder_indices = Data.get_token_level_weighted_encoder_indices(encoder_input_ids, high_comp_ratio, selected_token_indices, down_scaling_method)
                    lm_length = sum([len(x) for x in encoder_indices])
                    if high_comp_ratio > 40:
                        encoder_indices = Data.get_encoder_indices(encoder_input_ids, comp_ratio, down_scaling_method)
                        break
            else:
                encoder_indices = Data.get_encoder_indices(encoder_input_ids, comp_ratio, down_scaling_method)

            lm_length = sum([len(x) for x in encoder_indices])
            
            # * check
            if (len_for_encoder + 2 < lm_length)  and (low_ratio_index_length > 0):
                raise ValueErroe("Invalid encoder_indices")
                
            # * input_ids
            ph_indices_num = sum([len(x) for x in encoder_indices])
            input_ids = head_input_ids + normal_token_input_ids_pair[0] + [PH_TOKEN_ID] * ph_indices_num + normal_token_input_ids_pair[1] + tail_input_ids
            # * gen placeholder_indices
            ph_indices = [k + len(head_input_ids) + len(normal_token_input_ids_pair[0]) for k in range(ph_indices_num)]
            
            # * format
            outputs["input_ids"].append(input_ids)
            outputs["encoder_input_ids"].append(encoder_input_ids)
            outputs["ph_indices"].append(ph_indices)
            outputs["encoder_indices"].append(encoder_indices)

            # * if not eval, process labels
            if not eval_mode:
                head_labels = encoded_w_context["labels"][:len(head_input_ids)]
                tail_labels = encoded_w_context["labels"][-len(tail_input_ids):]
                labels = head_labels + [-100] * ph_indices_num + tail_labels

                outputs["labels"].append(labels)

            outputs["length"].append(len(input_ids))
            outputs["index"].append(indices[i])

        return outputs


    # embedding level selective compression
    def process_flexrag_embedding_sc(
        data: List[dict],
        indices: List[int],
        tokenizer: PreTrainedTokenizer,
        chat_template: str,
        lm_max_length: int,
        encoder_max_length: int,
        comp_candidates: List[int],
        dataset_name: str,
        text_proportion: float,
        importance_sentence_dict,
        low_comp_ratio: int=1,
        down_scaling_method: str="random",
        min_length: Optional[int]=None,
        max_length: Optional[int]=None,
        eval_mode: bool=False,
        ratio_power_of_two: bool=True,
        use_encoder_at_ratio_one: bool=False,
    ):
        outputs = {
            "input_ids": [],
            "encoder_input_ids": [],
            "ph_indices": [],
            "encoder_indices": [],
            "length": [],
            "index": [],
        }
        if not eval_mode:
            outputs["labels"] = []

        for i, conversations in enumerate(data["conversations"]):
            # * if eval, reformat conversations
            if eval_mode:
                conversations = [
                    conversations[0],
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
             # * cal compression_ratio
            if comp_ratio <= 0:
                remain_length = lm_max_length - len(head_input_ids) - len(tail_input_ids)
                if ratio_power_of_two:
                    comp_ratio = 2 ** math.ceil(math.log2(math.ceil(len(context_input_ids) / remain_length))) 
                else:
                    comp_ratio = math.ceil(len(context_input_ids) / remain_length)
            else:
                # * select compression ratio
                comp_ratio = comp_ratio

                # * truncate too long context
                max_encoder_token_num = (lm_max_length - len(head_input_ids) - len(tail_input_ids)) * comp_ratio
                max_encoder_token_num -= math.ceil(max_encoder_token_num / encoder_max_length) * special_token_num

                if (len(context_input_ids) > max_encoder_token_num) and not use_encoder_at_ratio_one:
                    half = max_et_token_num // 2
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

            # * Load prepared sentence information
            sentence_begin_indices = importance_sentence_dict[indices[i]]['sentence_begin_indices']
            sentence_priority_list = importance_sentence_dict[indices[i]]['sentence_priority_list']
            sentences_ids_list = importance_sentence_dict[indices[i]]['sentences_ids_list']

            # * Select specfic number of sentences based on the text proportion
            len_encoder_input_ids = 0
            for _encoder_input_ids in encoder_input_ids:
                len_encoder_input_ids += len(_encoder_input_ids)

            important_max_length = math.floor(len_encoder_input_ids * text_proportion)  # text propotion in original context 
            low_ratio_index_length = 0 
            sentences_length = 0
            selected_sentences_id = []
            
            for idx in sentence_priority_list:
                sentence_encoder_indices = Data.get_encoder_indices([context_input_ids[sentence_begin_indices[idx]['idx2']:sentence_begin_indices[idx + 1]['idx2']]], low_comp_ratio, down_scaling_method)
                if sentences_length + len(context_input_ids[sentence_begin_indices[idx]['idx2']:sentence_begin_indices[idx + 1]['idx2']]) > important_max_length:
                    continue

                low_ratio_index_length += len(sentence_encoder_indices[0])
                sentences_length += len(context_input_ids[sentence_begin_indices[idx]['idx2']:sentence_begin_indices[idx + 1]['idx2']]) 
                selected_sentences_id.append(idx)

            # * Based on the length of imporatance sentences after compress, calculate proper ratio for remain context
            if (len(context_input_ids) + len(head_input_ids) + len(tail_input_ids) - lm_max_length + 1) < 0:
                # open domain qa
                len_for_encoder = math.ceil(len(raw_encoder_input_ids) / comp_ratio)
                remain_length = len_for_encoder - low_ratio_index_length 
            else:
                # longbench    
                len_for_encoder = lm_max_length - len(head_input_ids) - len(tail_input_ids) - len(normal_token_input_ids_pair[0]) - len(normal_token_input_ids_pair[1])
                remain_length = len_for_encoder - low_ratio_index_length - len(selected_sentences_id) * 2 - len(encoder_input_ids)
            if remain_length < 0:
                raise ValueError("remain must be greater than 0. Consider lowering the low_comp_ratio or selection_propotion.")

            if ratio_power_of_two:
                high_comp_ratio = 2 ** math.ceil(math.log2(math.ceil((len_encoder_input_ids - sentences_length) / remain_length)))
            else:
                high_comp_ratio = math.ceil((len_encoder_input_ids - sentences_length) / remain_length)

            # * encoder_indices
            if low_ratio_index_length > 0:

                encoder_indices = Data.get_sentence_level_weighted_encoder_indices(encoder_input_ids, high_comp_ratio, low_comp_ratio, encoder_max_length, selected_sentences_id,  sentence_begin_indices, sentences_ids_list, down_scaling_method)

                # make sure that lm_length is less than len_for_encoder
                lm_length = sum([len(x) for x in encoder_indices])
                while lm_length > len_for_encoder:
                    high_comp_ratio += 1
                    encoder_indices = Data.get_sentence_level_weighted_encoder_indices(encoder_input_ids, high_comp_ratio, low_comp_ratio, encoder_max_length, selected_sentences_id,  sentence_begin_indices, sentences_ids_list, down_scaling_method)
                    lm_length = sum([len(x) for x in encoder_indices])
                    if high_comp_ratio > 100:
                        encoder_indices = Data.get_encoder_indices(encoder_input_ids, comp_ratio, down_scaling_method)
                        break
            else:
                encoder_indices = Data.get_encoder_indices(encoder_input_ids, comp_ratio, down_scaling_method)

            lm_length = sum([len(x) for x in encoder_indices])
            
            # * check
            if len_for_encoder < lm_length and low_ratio_index_length > 0:
                raise ValueErroe("Invalid encoder_indices")

            # * input_ids
            ph_indices_num = sum([len(x) for x in encoder_indices])
            input_ids = head_input_ids + normal_token_input_ids_pair[0] + [PH_TOKEN_ID] * ph_indices_num + normal_token_input_ids_pair[1] + tail_input_ids
            # * gen placeholder_indices
            ph_indices = [k + len(head_input_ids) + len(normal_token_input_ids_pair[0]) for k in range(ph_indices_num)]
            
            # * format
            outputs["input_ids"].append(input_ids)
            outputs["encoder_input_ids"].append(encoder_input_ids)
            outputs["ph_indices"].append(ph_indices)
            outputs["encoder_indices"].append(encoder_indices)

            # * if not eval, process labels
            if not eval_mode:
                head_labels = encoded_w_context["labels"][:len(head_input_ids)]
                tail_labels = encoded_w_context["labels"][-len(tail_input_ids):]
                labels = head_labels + [-100] * ph_indices_num + tail_labels
                outputs["labels"].append(labels)

            outputs["length"].append(len(input_ids))
            outputs["index"].append(indices[i])

        return outputs

    @staticmethod
    def _process_language_modeling(data, indices, tokenizer, eval_mode=False):
        pass

    @staticmethod
    def prepare_train_data(
        data_files: List[str],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        chat_template: Optional[str] = None,
        max_train_num_per_data: Optional[int] = None,
        window_mode: bool = False,
        lm_max_length: Optional[int] = None,
        encoder_max_length: Optional[int] = None,
        comp_candidates: Optional[List[int]] = None,
        load_from_disk: bool = False,
        seed: int = 42,
    ):
        # * fix random seed
        set_seed(seed)

        # * load dataset saved by `dataset.save_to_disk`, only pretrain data will load by this
        if load_from_disk:
            assert (
                len(data_files) == 1
            ), f"If `load_from_disk=True`, make sure your `data_files` only has one file. (data_files={data_files})"

            data_file = data_files[0]
            logger.info(f"Loading training dataset from {data_file}...")
            dataset = datasets.load_from_disk(data_file)
            if max_train_num_per_data:
                dataset = dataset.train_test_split(max_train_num_per_data, seed=seed)
                dataset = dataset["test"]
            return dataset

        # * load json format dataset
        train_datasets = []
        for data_file in data_files:
            dataset = datasets.load_dataset(
                    "json",
                    data_files=data_file,
                    split="train",
                )
            
            if "text" in dataset.column_names:
                process_fn = partial(
                    Data._process_language_modeling,
                    tokenizer=tokenizer,
                    min_length=min_length,
                    max_length=max_length,
                    eval_mode=False,
                )
            elif "conversations" in dataset.column_names:
                if window_mode:
                    process_fn = partial(
                        Data.process_instruction_tuning,
                        tokenizer=tokenizer,
                        chat_template=chat_template,
                        min_length=min_length,
                        max_length=max_length,
                        eval_mode=False,
                    )
                else:
                    assert (
                        lm_max_length is not None
                    ), "If `window_mode=False`, make sure `lm_max_length` is not None."
                    assert (
                        encoder_max_length is not None
                    ), "If `window_mode=False`, make sure `encoder_max_length` is not None."
                    assert (
                        comp_candidates is not None
                    ), "If `window_mode=False`, make sure `comp_candidates` is not None."

                    tokenizer.add_tokens([CONTEXT_TAG], special_tokens=True)

                    process_fn = partial(
                        Data.process_flexrag_instruction_tuning,
                        tokenizer=tokenizer,
                        chat_template=chat_template,
                        min_length=min_length,
                        max_length=max_length,
                        lm_max_length=lm_max_length,
                        encoder_max_length=encoder_max_length,
                        comp_candidates=comp_candidates,
                        eval_mode=False,
                    )
            else:
                raise ValueError(
                    "Training data's format is error. Can't find `text` or `conversations` in `column_names`."
                )

            dataset = dataset.map(
                process_fn,
                batched=True,
                num_proc=32,
                remove_columns=dataset.column_names,
                batch_size=32,
                with_indices=True,
                
            )
            if max_train_num_per_data and len(dataset) > max_train_num_per_data:
                dataset = dataset.train_test_split(max_train_num_per_data, seed=seed)[
                    "test"
                ]
            train_datasets.append(dataset)

        dataset = datasets.concatenate_datasets(train_datasets)
        return dataset


# * Colloator


@dataclass
class DefaultDataCollator:
    """
    Data collator that can:
    1. Dynamically pad all inputs received. The inputs must be dict of lists.
    2. Add position_ids based on attention_mask if required.
    """

    tokenizer: PreTrainedTokenizer
    attention_padding_value: int = 0
    label_padding_value: int = -100

    keys_to_tensorize = {
        "input_ids",
        "attention_mask",
        "labels",
        "position_ids",
        "token_type_ids",
        "length",
        "depth",
        "index",
    }

    def __call__(self, batch_elem: List) -> Dict[str, Any]:
        first_elem = batch_elem[0]
        return_batch = {}

        for key, value in first_elem.items():
            # HACK: any key containing attention_mask must be attention_mask
            # important to assign different pad token for different types of inputs
            if "attention_mask" in key:
                pad_token_id = self.attention_padding_value
            elif "label" in key:
                pad_token_id = self.label_padding_value
            else:
                pad_token_id = self.tokenizer.pad_token_id

            batch_value = [elem[key] for elem in batch_elem]
            # pad all lists and nested lists
            if isinstance(value, list) and key in self.keys_to_tensorize:
                max_length = get_max_length_in_nested_lists(batch_value)
                batch_value = pad_nested_lists(
                    batch_value, max_length, pad_token_id, self.tokenizer.padding_side
                )

            if key in self.keys_to_tensorize:
                return_batch[key] = torch.tensor(batch_value)
            else:
                # handle strings and None
                return_batch[key] = batch_value
        return return_batch


@dataclass
class FlexRAGCollator:
    tokenizer: PreTrainedTokenizer
    attention_padding_value: int = 0
    label_padding_value: int = -100

    def __call__(self, batch_elem: List) -> Dict[str, Any]:
        # * extract data from features, and format them from dict to list
        input_ids = [f["input_ids"] for f in batch_elem]  # List[List[int]]
        ph_indices = [f["ph_indices"] for f in batch_elem]  # List[List[int]]
        encoder_input_ids = [f["encoder_input_ids"] for f in batch_elem]  # List[List[List[int]]]
        encoder_indices = [f["encoder_indices"] for f in batch_elem]  # List[List[List[int]]]
        labels = (
            [f["labels"] for f in batch_elem] if "labels" in batch_elem[0] else None
        )  # List[List[int]]

        # * process model inputs
        input_ids, attention_mask, ph_indices, labels = self.process_model_inputs(
            input_ids, ph_indices, labels
        )

        # * process compressive encoder input
        encoder_input_ids, encoder_attention_mask, encoder_indices = self.process_encoder_inputs(
            encoder_input_ids, encoder_indices
        )

        # * to torch tensor
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        encoder_input_ids = torch.tensor(encoder_input_ids)
        encoder_attention_mask = torch.tensor(encoder_attention_mask)
        labels = torch.tensor(labels) if labels else None

        # * format
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ph_indices": ph_indices,
            "encoder_input_ids": encoder_input_ids,
            "encoder_attention_mask": encoder_attention_mask,
            "encoder_indices": encoder_indices,
            "labels": labels,
        }

    def process_model_inputs(self, input_ids, ph_indices, labels):
        # * get attention mask
        max_len = get_max_length_in_nested_lists(input_ids)
        attention_mask = get_attention_mask_from_nested_lists(input_ids)

        # * get new ph_indices since padding side is left
        ph_indices = [
            [idx + max_len - len(input_ids[i]) for idx in ph_indices[i]]
            for i in range(len(ph_indices))
        ]
        if sum([len(x) for x in ph_indices]) == 0:
            ph_indices = None

        # * pad
        input_ids = pad_nested_lists(
            input_ids, max_len, self.tokenizer.pad_token_id, "left"
        )
        attention_mask = pad_nested_lists(
            attention_mask, max_len, self.attention_padding_value, "left"
        )
        if labels:
            labels = pad_nested_lists(labels, max_len, self.label_padding_value, "left")

        return input_ids, attention_mask, ph_indices, labels

    def process_encoder_inputs(self, encoder_input_ids, encoder_indices):
        # * 3D -> 2D
        encoder_input_ids = sum(encoder_input_ids, [])  # List[List[int]]
        encoder_indices = sum(encoder_indices, [])  # List[List[int]]

        # * filter empty item
        new_encoder_input_ids = []
        new_encoder_indices = []
        for i in range(len(encoder_input_ids)):
            if len(encoder_indices[i]) != 0:
                new_encoder_input_ids.append(encoder_input_ids[i])
                new_encoder_indices.append(encoder_indices[i])
        encoder_input_ids = new_encoder_input_ids
        encoder_indices = new_encoder_indices

        if len(encoder_input_ids) == 0:
            return [], [], None

        # * get attention mask and pad
        max_len = get_max_length_in_nested_lists(encoder_input_ids)
        encoder_attention_mask = get_attention_mask_from_nested_lists(encoder_input_ids)

        encoder_indices = [
            [idx + max_len - len(encoder_input_ids[i]) for idx in encoder_indices[i]]
            for i in range(len(encoder_indices))
        ]

        encoder_input_ids = pad_nested_lists(
            encoder_input_ids, max_len, self.tokenizer.pad_token_id, "left",
        )
        encoder_attention_mask = pad_nested_lists(
            encoder_attention_mask, max_len, self.attention_padding_value, "left"
        )

        return encoder_input_ids, encoder_attention_mask, encoder_indices

# * Sampler
class StrideGroupedSampler(Sampler):
    def __init__(
        self,
        batch_size: int,
        window: int,
        dataset: Optional[Dataset] = None,
        lengths: Optional[List[int]] = None,
        model_input_name: Optional[str] = None,
    ):
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")

        if lengths is None:
            model_input_name = (
                model_input_name if model_input_name is not None else "input_ids"
            )
            if (
                not (
                    isinstance(dataset[0], dict)
                    or isinstance(dataset[0], BatchEncoding)
                )
                or model_input_name not in dataset[0]
            ):
                raise ValueError(
                    "Can only automatically infer lengths for datasets whose items are dictionaries with an "
                    f"'{model_input_name}' key."
                )
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                "If lengths is a torch.Tensor, LengthGroupedSampler will be slow. Converting lengths to List[int]..."
            )
            lengths = lengths.tolist()

        indices = list(range(len(lengths)))

        # * get number of strides for each data
        num_strides = []
        for length in lengths:
            num_stride = math.ceil(length / window)
            num_strides.append(num_stride)

        indice_stride_pairs = list(zip(indices, num_strides))
        # * shuffle the indices in advance, otherwise the randomness may be lost when all num_strides are equal
        random.shuffle(indice_stride_pairs)

        # * sort data according to the number of strides
        indice_stride_pairs = sorted(indice_stride_pairs, key=lambda x: x[1])

        # * group data instances with the same number of strides into the same batch
        batches = []
        batch = []
        prev_num_stride = None
        for index, num_stride in indice_stride_pairs:
            if num_stride != prev_num_stride:
                batch.clear()

            batch.append(index)
            prev_num_stride = num_stride

            if len(batch) == batch_size:
                batches.append((batch.copy(), num_stride))
                batch.clear()

        random.shuffle(batches)

        batches = [x[0] for x in batches]
        self.indices = sum(batches, [])

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)


# * Utilities


def get_max_length_in_nested_lists(lst):
    if isinstance(lst[0], list):
        lengths = []
        for elem in lst:
            length = get_max_length_in_nested_lists(elem)
            lengths.append(length)
        max_length = max(lengths)
        return max_length
    else:
        return len(lst)


def get_attention_mask_from_nested_lists(lst):
    if isinstance(lst[0], list):
        attention_mask = []
        for elem in lst:
            mask = get_attention_mask_from_nested_lists(elem)
            attention_mask.append(mask)
        return attention_mask
    else:
        return [1] * len(lst)


def pad_nested_lists(lst, max_length, padding_value, padding_side="right"):
    if isinstance(lst, list) and len(lst) and isinstance(lst[0], list):
        for i, elem in enumerate(lst):
            lst[i] = pad_nested_lists(elem, max_length, padding_value, padding_side)
        return lst
    elif isinstance(lst, list):
        if padding_side == "right":
            return lst + [padding_value for _ in range(max_length - len(lst))]
        else:
            return [padding_value for _ in range(max_length - len(lst))] + lst
    else:
        raise NotImplementedError(f"Unrecognized type {lst}")
