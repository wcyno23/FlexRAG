import torch
from llmlingua import PromptCompressor
from typing import List
class MyCompressor(PromptCompressor):
    def __init__(
        self,
        model_name: str = "NousResearch/Llama-2-7b-hf",
        device_map: str = "cuda",
        model_config: dict = {},
        open_api_config: dict = {},
        use_llmlingua2: bool = False,
        llmlingua2_config: dict = {},
    ):
        PromptCompressor.__init__(self, model_name, device_map, model_config, open_api_config, use_llmlingua2, llmlingua2_config)

    def compress_prompt(
        self,
        context: List[str],
        instruction: str = "",
        question: str = "",
        rate: float = 0.5,
        target_token: float = -1,
        iterative_size: int = 200,
        force_context_ids: List[int] = None,
        force_context_number: int = None,
        use_sentence_level_filter: bool = False,
        use_context_level_filter: bool = True,
        use_token_level_filter: bool = True,
        keep_split: bool = False,
        keep_first_sentence: int = 0,
        keep_last_sentence: int = 0,
        keep_sentence_number: int = 0,
        high_priority_bonus: int = 100,
        context_budget: str = "+100",
        token_budget_ratio: float = 1.4,
        condition_in_question: str = "none",
        reorder_context: str = "original",
        dynamic_context_compression_ratio: float = 0.0,
        condition_compare: bool = False,
        add_instruction: bool = False,
        rank_method: str = "llmlingua",
        concate_question: bool = True,
        context_segs: List[str] = None,
        context_segs_rate: List[float] = None,
        context_segs_compress: List[bool] = None,
        target_context: int = -1,
        context_level_rate: float = 1.0,
        context_level_target_token: int = -1,
        return_word_label: bool = False,
        word_sep: str = "\t\t|\t\t",
        label_sep: str = " ",
        token_to_word: str = "mean",
        force_tokens: List[str] = [],
        force_reserve_digit: bool = False,
        drop_consecutive: bool = False,
        chunk_end_tokens: List[str] = [".", "\n"],
        strict_preserve_uncompressed: bool = True,
    ):
        """
        Compresses the given context.

        Args:
            context (List[str]): List of context strings that form the basis of the prompt.
            instruction (str, optional): Additional instruction text to be included in the prompt. Default is an empty string.
            question (str, optional): A specific question that the prompt is addressing. Default is an empty string.
            rate (float, optional): The maximum compression rate target to be achieved. The compression rate is defined
                the same as in paper "Language Modeling Is Compression". Delétang, Grégoire, Anian Ruoss, Paul-Ambroise Duquenne,
                Elliot Catt, Tim Genewein, Christopher Mattern, Jordi Grau-Moya et al. "Language modeling is compression."
                arXiv preprint arXiv:2309.10668 (2023):
                .. math::\text{Compression Rate} = \frac{\text{Compressed Size}}{\text{Raw Size}}
                Default is 0.5. The actual compression rate is generally lower than the specified target, but there can be
                fluctuations due to differences in tokenizers. If specified, it should be a float less than or equal
                to 1.0, representing the target compression rate.
            target_token (float, optional): The maximum number of tokens to be achieved. Default is -1, indicating no specific target.
                The actual number of tokens after compression should generally be less than the specified target_token, but there can
                be fluctuations due to differences in tokenizers. If specified, compression will be based on the target_token as
                the sole criterion, overriding the ``rate``.
            iterative_size (int, optional): The number of tokens to consider in each iteration of compression. Default is 200.
            force_context_ids (List[int], optional): List of specific context IDs to always include in the compressed result. Default is None.
            force_context_number (int, optional): The number of context sections to forcibly include. Default is None.
            use_sentence_level_filter (bool, optional): Whether to apply sentence-level filtering in compression. Default is False.
            use_context_level_filter (bool, optional): Whether to apply context-level filtering in compression. Default is True.
            use_token_level_filter (bool, optional): Whether to apply token-level filtering in compression. Default is True.
            keep_split (bool, optional): Whether to preserve the original separators without compression. Default is False.
            keep_first_sentence (int, optional): Number of sentences to forcibly preserve from the start of the context. Default is 0.
            keep_last_sentence (int, optional): Number of sentences to forcibly preserve from the end of the context. Default is 0.
            keep_sentence_number (int, optional): Total number of sentences to forcibly preserve in the compression. Default is 0.
            high_priority_bonus (int, optional): Bonus score for high-priority sentences to influence their likelihood of being retained. Default is 100.
            context_budget (str, optional): Token budget for the context-level filtering, expressed as a string to indicate flexibility. Default is "+100".
            token_budget_ratio (float, optional): Ratio to adjust token budget during sentence-level filtering. Default is 1.4.
            condition_in_question (str, optional): Specific condition to apply to question in the context. Default is "none".
            reorder_context (str, optional): Strategy for reordering context in the compressed result. Default is "original".
            dynamic_context_compression_ratio (float, optional): Ratio for dynamically adjusting context compression. Default is 0.0.
            condition_compare (bool, optional): Whether to enable condition comparison during token-level compression. Default is False.
            add_instruction (bool, optional): Whether to add the instruction to the prompt prefix. Default is False.
            rank_method (str, optional): Method used for ranking elements during compression. Default is "llmlingua".
            concate_question (bool, optional): Whether to concatenate the question to the compressed prompt. Default is True.

            target_context (int, optional): The maximum number of contexts to be achieved. Default is -1, indicating no specific target.
            context_level_rate (float, optional): The minimum compression rate target to be achieved in context level. Default is 1.0.
            context_level_target_token (float, optional): The maximum number of tokens to be achieved in context level compression.
                Default is -1, indicating no specific target. Only used in the coarse-to-fine compression senario.
            force_context_ids (List[int], optional): List of specific context IDs to always include in the compressed result. Default is None.
            return_word_label (bool, optional): Whether to return word with corresponding label. Default is False.
            word_sep (str, optional): The sep token used in fn_labeled_original_prompt to partition words. Default is "\t\t|\t\t".
            label_sep (str, optional): The sep token used in fn_labeled_original_prompt to partition word and label.  Default is " ".
            token_to_word (str, optional): How to convert token probability to word probability. Default is "mean".
            force_tokens (List[str], optional): List of specific tokens to always include in the compressed result. Default is [].
            force_reserve_digit  (bool, optional): Whether to forcibly reserve tokens that containing digit (0,...,9). Default is False.
            drop_consecutive (bool, optinal): Whether to drop tokens which are in 'force_tokens' but appears consecutively in compressed prompt.
                Default is False.
            chunk_end_tokens (List[str], optinal): The early stop tokens for segmenting chunk. Default is [".", "\n"],
        Returns:
            dict: A dictionary containing:
                - "compressed_prompt" (str): The resulting compressed prompt.
                - "compressed_prompt_list" (List[str]): List of the resulting compressed prompt. Only used in llmlingua2.
                - "fn_labeled_original_prompt" (str): original words along with their labels
                    indicating whether to reserve in compressed prompt, in the format (word label_sep label)
                    Only used in llmlingua2 when return_word_label = True.
                - "origin_tokens" (int): The original number of tokens in the input.
                - "compressed_tokens" (int): The number of tokens in the compressed output.
                - "ratio" (str): The compression ratio achieved, calculated as the original token number divided by the token number after compression.
                - "rate" (str): The compression rate achieved, in a human-readable format.
                - "saving" (str): Estimated savings in GPT-4 token usage.
        """
        if self.use_llmlingua2:
            return self.compress_prompt_llmlingua2(
                context,
                rate=rate,
                target_token=target_token,
                use_context_level_filter=use_context_level_filter,
                use_token_level_filter=use_token_level_filter,
                target_context=target_context,
                context_level_rate=context_level_rate,
                context_level_target_token=context_level_target_token,
                force_context_ids=force_context_ids,
                return_word_label=return_word_label,
                word_sep=word_sep,
                label_sep=label_sep,
                token_to_word=token_to_word,
                force_tokens=force_tokens,
                force_reserve_digit=force_reserve_digit,
                drop_consecutive=drop_consecutive,
                chunk_end_tokens=chunk_end_tokens,
            )
        assert (
            rate <= 1.0
        ), "Error: 'rate' must not exceed 1.0. The value of 'rate' indicates compression rate and must be within the range [0, 1]."

        if not context:
            context = [" "]
        if isinstance(context, str):
            context = [context]
        assert not (
            rank_method == "longllmlingua" and not question
        ), "In the LongLLMLingua, it is necessary to set a question."
        if condition_compare and "_condition" not in condition_in_question:
            condition_in_question += "_condition"
        if rank_method == "longllmlingua":
            if condition_in_question == "none":
                condition_in_question = "after"
        elif rank_method == "llmlingua":
            condition_in_question = (
                "none"
                if "_condition" not in condition_in_question
                else "none_condition"
            )
        origin_tokens = len(
            self.oai_tokenizer.encode(
                "\n\n".join([instruction] + context + [question]).strip()
            )
        )
        
        torch.set_printoptions(threshold=torch.inf)

        # for extracting
        original_token_id = torch.tensor(self.tokenizer.encode(context[0]))
        context_tokens_length = [self.get_token_length(c) for c in context]
        instruction_tokens_length, question_tokens_length = self.get_token_length(
            instruction
        ), self.get_token_length(question)
        if target_token == -1:
            target_token = (
                (
                    instruction_tokens_length
                    + question_tokens_length
                    + sum(context_tokens_length)
                )
                * rate
                - instruction_tokens_length
                - (question_tokens_length if concate_question else 0)
            )
        condition_flag = "_condition" in condition_in_question
        condition_in_question = condition_in_question.replace("_condition", "")

        if len(context) > 1 and use_context_level_filter:
            context, dynamic_ratio, context_used = self.control_context_budget(
                context,
                context_tokens_length,
                target_token,
                force_context_ids,
                force_context_number,
                question,
                condition_in_question,
                reorder_context=reorder_context,
                dynamic_context_compression_ratio=dynamic_context_compression_ratio,
                rank_method=rank_method,
                context_budget=context_budget,
                context_segs=context_segs,
                context_segs_rate=context_segs_rate,
                context_segs_compress=context_segs_compress,
                strict_preserve_uncompressed=strict_preserve_uncompressed,
            )
            if context_segs is not None:
                context_segs = [context_segs[idx] for idx in context_used]
                context_segs_rate = [context_segs_rate[idx] for idx in context_used]
                context_segs_compress = [
                    context_segs_compress[idx] for idx in context_used
                ]
        else:
            dynamic_ratio = [0.0] * len(context)

        segments_info = []
        if use_sentence_level_filter:
            context, segments_info = self.control_sentence_budget(
                context,
                target_token,
                keep_first_sentence=keep_first_sentence,
                keep_last_sentence=keep_last_sentence,
                keep_sentence_number=keep_sentence_number,
                high_priority_bonus=high_priority_bonus,
                token_budget_ratio=token_budget_ratio,
                question=question,
                condition_in_question=condition_in_question,
                rank_method=rank_method,
                context_segs=context_segs,
                context_segs_rate=context_segs_rate,
                context_segs_compress=context_segs_compress,
            )
        elif context_segs is not None:
            for context_idx in range(len(context)):
                segments_info.append(
                    [
                        (len(seg_text), seg_rate, seg_compress)
                        for seg_text, seg_rate, seg_compress in zip(
                            context_segs[context_idx],
                            context_segs_rate[context_idx],
                            context_segs_compress[context_idx],
                        )
                    ]
                )
        segments_info = [
            self.concate_segment_info(segment_info) for segment_info in segments_info
        ]

        if condition_flag:
            prefix = question + "\n\n" + instruction if add_instruction else question
            if (
                self.get_token_length(prefix + "\n\n") + iterative_size * 2
                > self.max_position_embeddings
            ):
                tokens = self.tokenizer(prefix, add_special_tokens=False).input_ids
                prefix = self.tokenizer.decode(
                    tokens[: self.prefix_bos_num]
                    + tokens[
                        len(tokens)
                        - self.max_position_embeddings
                        + 2
                        + self.prefix_bos_num
                        + 2 * iterative_size :
                    ]
                )
            start = self.get_prefix_length(prefix + "\n\n", context[0])
            context = [prefix] + context
        else:
            start = 0

        if use_token_level_filter:
            context = self.iterative_compress_prompt(
                context,
                target_token,
                iterative_size=iterative_size,
                keep_split=keep_split,
                start=start,
                dynamic_ratio=dynamic_ratio,
                condition_compare=condition_compare,
                segments_info=segments_info,
            )
            compressed_prompt = (
                self.tokenizer.batch_decode(context[0])[0]
                .replace("<s> ", "")
                .replace("<s>", "")
            )
        else:
            if condition_flag:
                context = context[1:]
            compressed_prompt = "\n\n".join(context)

        res = []
        if instruction:
            res.append(instruction)
        if compressed_prompt.strip():
            res.append(compressed_prompt)
        if question and concate_question:
            res.append(question)
        
        # original one
        # compressed_prompt = "\n\n".join(res)
        
        # mine
        compressed_prompt = "".join(res)

        compressed_tokens = len(self.oai_tokenizer.encode(compressed_prompt))
        saving = (origin_tokens - compressed_tokens) * 0.06 / 1000
        ratio = 1 if compressed_tokens == 0 else origin_tokens / compressed_tokens
        rate = 1 / ratio
        token_idx = get_token_idx(original_token_id, context[0][0])
        return {
            "compressed_prompt": compressed_prompt,
            "origin_tokens": origin_tokens,
            "compressed_tokens": compressed_tokens,
            "ratio": f"{ratio:.1f}x",
            "rate": f"{rate * 100:.1f}%",
            "saving": f", Saving ${saving:.1f} in GPT-4.",
            "context_id": context[0],
            "token_idx": token_idx,
        }

    def compress_prompt_llmlingua2(
        self,
        context: List[str],
        rate: float = 0.5,
        target_token: int = -1,
        use_context_level_filter: bool = False,
        use_token_level_filter: bool = True,
        target_context: int = -1,
        context_level_rate: float = 1.0,
        context_level_target_token: int = -1,
        force_context_ids: List[int] = [],
        return_word_label: bool = False,
        word_sep: str = "\t\t|\t\t",
        label_sep: str = " ",
        token_to_word: str = "mean",
        force_tokens: List[str] = [],
        force_reserve_digit: bool = False,
        drop_consecutive: bool = False,
        chunk_end_tokens: List[str] = [".", "\n"],
    ):
        """
        Compresses the given context, instruction and question.

        Args:
            context (List[str]): List of context strings that form the basis of the prompt.
            rate (float, optional): The minimum compression rate target to be achieved. Default is 0.5. The actual compression rate
                generally exceeds the specified target, but there can be fluctuations due to differences in tokenizers. If specified,
                it should be a float greater than or equal to 1.0, representing the target compression rate.
            target_token (int, optional): The maximum number of tokens to be achieved. Default is -1, indicating no specific target.
                The actual number of tokens after compression should generally be less than the specified target_token, but there can
                be fluctuations due to differences in tokenizers. If specified, compression will be based on the target_token as
                the sole criterion, overriding the rate.
            target_context (int, optional): The maximum number of contexts to be achieved. Default is -1, indicating no specific target.
                Only used in the coarse-to-fine compression.
            context_level_rate (float, optional): The minimum compression rate target to be achieved in context level. Default is 1.0.
                Only used in the coarse-to-fine compression.
            context_level_target_token (float, optional): The maximum number of tokens to be achieved in context level compression.
                Default is -1, indicating no specific target. Only used in the coarse-to-fine compression senario.
            force_context_ids (List[int], optional): List of specific context IDs to always include in the compressed result. Default is None.
            return_word_label (bool, optional): Whether to return word with corresponding label. Default is False.
            word_sep (str, optional): The sep token used in fn_labeled_original_prompt to partition words. Default is "\t\t|\t\t".
            label_sep (str, optional): The sep token used in fn_labeled_original_prompt to partition word and label.  Default is " ".
            token_to_word (str, optional): How to convert token probability to word probability. Default is "mean".
            force_tokens (List[str], optional): List of specific tokens to always include in the compressed result. Default is [].
            force_reserve_digit  (bool, optional): Whether to forcibly reserve tokens that containing digit (0,...,9). Default is False.
            drop_consecutive (bool, optinal): Whether to drop tokens which are in 'force_tokens' but appears consecutively in compressed prompt.
                Default is False.
            chunk_end_tokens (List[str], optional): The early stop tokens for segmenting chunk. Default is [".", "\n"].
        Returns:
            dict: A dictionary containing:
                - "compressed_prompt" (str): The resulting compressed prompt.
                - "compressed_prompt_list" (List[str]): List of the resulting compressed prompt.
                - "fn_labeled_original_prompt" (str): original words along with their labels
                    indicating whether to reserve in compressed prompt, in the format (word label_sep label)
                - "origin_tokens" (int): The original number of tokens in the input.
                - "compressed_tokens" (int): The number of tokens in the compressed output.
                - "ratio" (str): The compression ratio achieved, in a human-readable format.
                - "rate" (str): The compression rate achieved, in a human-readable format.
                - "saving" (str): Estimated savings in GPT-4 token usage.

        """
        assert len(force_tokens) <= self.max_force_token
        token_map = {}
        for i, t in enumerate(force_tokens):
            if len(self.tokenizer.tokenize(t)) != 1:
                token_map[t] = self.added_tokens[i]
        chunk_end_tokens = copy.deepcopy(chunk_end_tokens)
        for c in chunk_end_tokens:
            if c in token_map:
                chunk_end_tokens.append(token_map[c])
        chunk_end_tokens = set(chunk_end_tokens)

        if type(context) == str:
            context = [context]
        context = copy.deepcopy(context)

        if len(context) == 1 and use_context_level_filter:
            use_context_level_filter = False

        n_original_token = 0
        context_chunked = []
        for i in range(len(context)):
            n_original_token += self.get_token_length(
                context[i], use_oai_tokenizer=True
            )
            for ori_token, new_token in token_map.items():
                context[i] = context[i].replace(ori_token, new_token)
            context_chunked.append(
                self.__chunk_context(context[i], chunk_end_tokens=chunk_end_tokens)
            )

        if use_context_level_filter:
            # want use_context_level_filter but do not specify any parameters in context level?
            # we will set context_level_rate = (rate + 1.0) / 2 if specify rate or target_token * 2 if specify target_token
            if (
                target_context <= 0
                and context_level_rate >= 1.0
                and context_level_target_token <= 0
            ):
                if target_token < 0 and rate < 1.0:
                    context_level_rate = (
                        (rate + 1.0) / 2 if use_token_level_filter else rate
                    )
                if target_token >= 0:
                    context_level_target_token = (
                        target_token * 2 if use_token_level_filter else target_token
                    )

            if target_context >= 0:
                context_level_rate = min(target_context / len(context), 1.0)
            if context_level_target_token >= 0:
                context_level_rate = min(
                    context_level_target_token / n_original_token, 1.0
                )

            context_probs, context_words = self.__get_context_prob(
                context_chunked,
                token_to_word=token_to_word,
                force_tokens=force_tokens,
                token_map=token_map,
                force_reserve_digit=force_reserve_digit,
            )

            threshold = np.percentile(
                context_probs, int(100 * (1 - context_level_rate))
            )

            reserved_context = []
            context_label = [False] * len(context_probs)
            for i, p in enumerate(context_probs):
                if p >= threshold or (
                    force_context_ids is not None and i in force_context_ids
                ):
                    reserved_context.append(context_chunked[i])
                    context_label[i] = True
            n_reserved_token = 0
            for chunks in reserved_context:
                for c in chunks:
                    n_reserved_token += self.get_token_length(c, use_oai_tokenizer=True)
            if target_token >= 0:
                rate = min(target_token / n_reserved_token, 1.0)

            if use_token_level_filter:
                compressed_context, word_list, word_label_list = self.__compress(
                    reserved_context,
                    reduce_rate=max(0, 1 - rate),
                    token_to_word=token_to_word,
                    force_tokens=force_tokens,
                    token_map=token_map,
                    force_reserve_digit=force_reserve_digit,
                    drop_consecutive=drop_consecutive,
                )
            else:
                compressed_context, word_list, word_label_list = self.__compress(
                    reserved_context,
                    reduce_rate=0,
                    token_to_word=token_to_word,
                    force_tokens=force_tokens,
                    token_map=token_map,
                    force_reserve_digit=force_reserve_digit,
                    drop_consecutive=drop_consecutive,
                )

            n_compressed_token = 0
            for c in compressed_context:
                n_compressed_token += self.get_token_length(c, use_oai_tokenizer=True)
            saving = (n_original_token - n_compressed_token) * 0.06 / 1000
            ratio = (
                1 if n_compressed_token == 0 else n_original_token / n_compressed_token
            )
            res = {
                "compressed_prompt": "\n\n".join(compressed_context),
                "compressed_prompt_list": compressed_context,
                "origin_tokens": n_original_token,
                "compressed_tokens": n_compressed_token,
                "ratio": f"{ratio:.1f}x",
                "rate": f"{1 / ratio * 100:.1f}%",
                "saving": f", Saving ${saving:.1f} in GPT-4.",
            }
            if return_word_label:
                words = []
                labels = []
                j = 0
                for i in range(len(context)):
                    if context_label[i]:
                        words.extend(word_list[j])
                        labels.extend(word_label_list[j])
                        j += 1
                    else:
                        words.extend(context_words[i])
                        labels.extend([0] * len(context_words[i]))
                word_label_lines = word_sep.join(
                    [f"{word}{label_sep}{label}" for word, label in zip(words, labels)]
                )
                res["fn_labeled_original_prompt"] = word_label_lines
            return res

        if target_token > 0:
            rate = min(target_token / n_original_token, 1.0)

        if use_token_level_filter:
            compressed_context, word_list, word_label_list = self.__compress(
                context_chunked,
                reduce_rate=max(0, 1 - rate),
                token_to_word=token_to_word,
                force_tokens=force_tokens,
                token_map=token_map,
                force_reserve_digit=force_reserve_digit,
                drop_consecutive=drop_consecutive,
            )
        else:
            compressed_context, word_list, word_label_list = self.__compress(
                context_chunked,
                reduce_rate=0,
                token_to_word=token_to_word,
                force_tokens=force_tokens,
                token_map=token_map,
                force_reserve_digit=force_reserve_digit,
                drop_consecutive=drop_consecutive,
            )

        n_compressed_token = 0
        for c in compressed_context:
            n_compressed_token += self.get_token_length(c, use_oai_tokenizer=True)
        saving = (n_original_token - n_compressed_token) * 0.06 / 1000
        ratio = 1 if n_compressed_token == 0 else n_original_token / n_compressed_token
        res = {
            "compressed_prompt": "\n\n".join(compressed_context),
            "compressed_prompt_list": compressed_context,
            "origin_tokens": n_original_token,
            "compressed_tokens": n_compressed_token,
            "ratio": f"{ratio:.1f}x",
            "rate": f"{1 / ratio * 100:.1f}%",
            "saving": f", Saving ${saving:.1f} in GPT-4.",
        }
        if return_word_label:
            words = []
            labels = []
            for w_list, l_list in zip(word_list, word_label_list):
                words.extend(w_list)
                labels.extend(l_list)

            word_label_lines = word_sep.join(
                [f"{word}{label_sep}{label}" for word, label in zip(words, labels)]
            )
            res["fn_labeled_original_prompt"] = word_label_lines
        return res

    def iterative_compress_prompt(
        self,
        context: List[str],
        target_token: float,
        iterative_size: int = 200,
        keep_split: bool = False,
        split_token_id: int = 13,
        start: int = 0,
        dynamic_ratio: list = None,
        condition_compare: bool = False,
        segments_info: List[List[tuple]] = None,
    ):
        if segments_info is None or segments_info == []:
            iterative_ratios = self.get_dynamic_compression_ratio(
                context, target_token, iterative_size, dynamic_ratio, start
            )
        else:
            iterative_ratios = self.get_structured_dynamic_compression_ratio(
                context, iterative_size, dynamic_ratio, start, segments_info
            )
        context = "\n\n".join(context)
        tokenized_text = self.tokenizer(
            context, return_tensors="pt", add_special_tokens=False
        )
        input_ids = tokenized_text["input_ids"].to(self.device)
        attention_mask = tokenized_text["attention_mask"].to(self.device)

        N = (attention_mask == 1).sum()
        compressed_input_ids, compressed_attention_mask = input_ids, attention_mask
        if condition_compare:
            self_input_ids, self_attention_mask = (
                input_ids[:, start:],
                attention_mask[:, start:],
            )
            self_compressed_input_ids, self_compressed_attention_mask = (
                self_input_ids,
                self_attention_mask,
            )

        end = min(iterative_size + start, compressed_input_ids.shape[1])
        threshold, keep_flag = None, None
        if keep_split:
            input_ids_numpy = input_ids.cpu().detach().numpy()[0]
            N = len(input_ids_numpy)
            keep_flag = [
                int(
                    (
                        ii > 0
                        and input_ids_numpy[ii] == split_token_id
                        and input_ids_numpy[ii - 1] == split_token_id
                    )
                    or (
                        ii < N - 1
                        and input_ids_numpy[ii] == split_token_id
                        and input_ids_numpy[ii + 1] == split_token_id
                    )
                )
                for ii in range(N)
            ]
            keep_flag = torch.tensor(keep_flag).to(self.device)
        past_key_values, past_loss, ready_end = None, None, 0
        self_past_key_values, self_past_loss, self_ready_end = None, None, 0
        pop_compressed_input_ids, pop_self_compressed_input_ids = None, None
        idx = 0
        # need
        need_idx = None
        while end <= compressed_input_ids.shape[1]:
            if end > self.max_position_embeddings and past_key_values is not None:
                # KV-Cache Compression
                e, s = end - self.max_position_embeddings, min(
                    self.cache_bos_num + start, self.max_position_embeddings
                )
                if pop_compressed_input_ids is None:
                    pop_compressed_input_ids = compressed_input_ids[:, :e]
                else:
                    pop_compressed_input_ids = torch.cat(
                        [pop_compressed_input_ids, compressed_input_ids[:, :e]], dim=-1
                    )
                compressed_input_ids = compressed_input_ids[:, e:]
                compressed_attention_mask = compressed_attention_mask[:, e:]
                past_key_values = [
                    [
                        torch.cat([k[..., :s, :], k[..., s + e :, :]], dim=-2),
                        torch.cat([v[..., :s, :], v[..., s + e :, :]], dim=-2),
                    ]
                    for k, v in past_key_values
                ]
                if keep_flag is not None:
                    keep_flag = keep_flag[e:]
                end, ready_end = end - e, ready_end - e
                if condition_compare:
                    s = min(s, self_past_key_values[0][0].shape[2] - e)
                    self_ready_end -= e
                    if pop_self_compressed_input_ids is None:
                        pop_self_compressed_input_ids = self_compressed_input_ids[:, :e]
                    else:
                        pop_self_compressed_input_ids = torch.cat(
                            [
                                pop_self_compressed_input_ids,
                                self_compressed_input_ids[:, :e],
                            ],
                            dim=-1,
                        )
                    self_compressed_input_ids = self_compressed_input_ids[:, e:]
                    self_compressed_attention_mask = self_compressed_attention_mask[
                        :, e:
                    ]
                    self_past_key_values = [
                        [
                            torch.cat([k[..., :s, :], k[..., s + e :, :]], dim=-2),
                            torch.cat([v[..., :s, :], v[..., s + e :, :]], dim=-2),
                        ]
                        for k, v in self_past_key_values
                    ]

            loss, past_key_values = self.get_ppl(
                "",
                "token",
                compressed_input_ids,
                compressed_attention_mask,
                past_key_values=past_key_values,
                return_kv=True,
                end=end if idx else None,
            )
            if loss.shape[0] == 0:
                break
            if past_loss is not None:
                if end - 1 > len(past_loss):
                    past_loss = torch.cat(
                        [past_loss, torch.zeros_like(loss)[: end - 1 - len(past_loss)]]
                    )
                past_loss[ready_end : end - 1] = loss
                loss = past_loss
            else:
                past_loss = loss
            if idx:
                past_key_values = [
                    [k[:, :, : end - iterative_size], v[:, :, : end - iterative_size]]
                    for k, v in past_key_values
                ]
            else:
                past_key_values = None

            if condition_compare:
                self_loss, self_past_key_values = self.get_ppl(
                    "",
                    "token",
                    self_compressed_input_ids,
                    self_compressed_attention_mask,
                    past_key_values=self_past_key_values,
                    return_kv=True,
                    end=end - start if idx else None,
                )
                if self_past_loss is not None:
                    if end - start - 1 > len(self_past_loss):
                        self_past_loss = torch.cat(
                            [
                                self_past_loss,
                                torch.zeros_like(self_loss)[
                                    : end - 1 - start - len(self_past_loss)
                                ],
                            ]
                        )
                    self_past_loss[self_ready_end : end - start - 1] = self_loss
                    self_loss = self_past_loss
                else:
                    self_past_loss = self_loss
                if idx:
                    self_past_key_values = [
                        [
                            k[:, :, : end - iterative_size - start],
                            v[:, :, : end - iterative_size - start],
                        ]
                        for k, v in self_past_key_values
                    ]
                else:
                    self_past_key_values = None

                self_ready_end = (
                    end - start - iterative_size if not (start and idx == 0) else 0
                )
            ready_end = end - iterative_size if not (start and idx == 0) else 0

            for delta_end, ratio in iterative_ratios[idx]:
                loss = past_loss
                if condition_compare:
                    self_loss = self_past_loss
                    threshold = self.get_estimate_threshold_base_distribution(
                        self_loss[: loss[start:].shape[0]] - loss[start:], ratio, False
                    )
                else:
                    threshold = self.get_estimate_threshold_base_distribution(
                        loss, ratio, False
                    )

                (
                    compressed_input_ids,
                    compressed_attention_mask,
                    keep_flag,
                    end,
                    past_loss,
                    self_past_loss,
                    self_compressed_input_ids,
                    self_compressed_attention_mask,
                    # 
                    need_idx_temp,
                ) = self.get_compressed_input(
                    loss,
                    compressed_input_ids,
                    compressed_attention_mask,
                    end - iterative_size + delta_end,
                    iterative_size=delta_end,
                    threshold=threshold,
                    keep_flag=keep_flag,
                    split_token_id=split_token_id,
                    start=start,
                    self_loss=self_loss if condition_compare else None,
                    self_input_ids=(
                        self_compressed_input_ids if condition_compare else None
                    ),
                    self_attention_mask=(
                        self_compressed_attention_mask if condition_compare else None
                    ),
                )

                # extract need idx
                if need_idx is None:
                    need_idx = need_idx_temp
                else:
                    need_idx = torch.cat([need_idx, need_idx_temp])

                end += iterative_size
            idx += 1
        if pop_compressed_input_ids is not None:
            compressed_input_ids = torch.cat(
                [pop_compressed_input_ids, compressed_input_ids], dim=-1
            )
        return compressed_input_ids[:, start:], compressed_attention_mask[:, start:]

    def get_compressed_input(
        self,
        loss,
        input_ids,
        attention_mask,
        end=200,
        iterative_size=200,
        threshold=0.5,
        keep_flag=None,
        split_token_id: int = 13,
        start: int = 0,
        self_loss=None,
        self_input_ids=None,
        self_attention_mask=None,
    ):
        if self_loss is not None:
            need_idx = torch.concat(
                [
                    loss[:start] > 0,
                    self_loss[: loss[start:].shape[0]] - loss[start:] > threshold,
                    loss[:1] > 0,
                ]
            )
        else:
            need_idx = torch.concat([loss > threshold, loss[:1] > 0])
        need_idx[end:] = 1
        need_idx[: end - iterative_size] = 1
        loss = loss[need_idx[:-1]]
        if self_loss is not None:
            if need_idx.shape[0] < self_loss.shape[0] + start + 1:
                need_idx = torch.cat(
                    [
                        need_idx,
                        torch.ones(
                            self_loss.shape[0] - need_idx.shape[0] + start + 1,
                            dtype=torch.bool,
                        ).to(need_idx.device),
                    ]
                )
            self_loss = self_loss[need_idx[start:-1]]

        if need_idx.shape[0] < input_ids.shape[1]:
            need_idx = torch.cat(
                [
                    need_idx,
                    torch.ones(
                        input_ids.shape[1] - need_idx.shape[0], dtype=torch.bool
                    ).to(need_idx.device),
                ]
            )
        elif need_idx.shape[0] > input_ids.shape[1]:
            need_idx = need_idx[: input_ids.shape[1]]

        if keep_flag is not None:
            need_idx[keep_flag == 1] = 1
        last = -1
        if keep_flag is not None:
            for ii in range(max(0, end - iterative_size), end):
                if need_idx[ii] != 1:
                    continue
                now = input_ids[0][ii].detach().cpu().item()
                if (
                    now == split_token_id
                    and last == split_token_id
                    and keep_flag[ii].detach().cpu().item() == 0
                ):
                    need_idx[ii] = 0
                else:
                    last = now
        compressed_input_ids = input_ids[attention_mask == 1][need_idx].unsqueeze(0)
        compressed_attention_mask = attention_mask[attention_mask == 1][
            need_idx
        ].unsqueeze(0)

        if self_loss is not None:
            self_compressed_input_ids = self_input_ids[self_attention_mask == 1][
                need_idx[start:]
            ].unsqueeze(0)
            self_compressed_attention_mask = self_attention_mask[
                self_attention_mask == 1
            ][need_idx[start:]].unsqueeze(0)
        else:
            self_compressed_input_ids, self_compressed_attention_mask = None, None
        if keep_flag is not None:
            if len(keep_flag) > len(need_idx):
                keep_flag = torch.cat(
                    [
                        keep_flag[:start],
                        keep_flag[start : len(need_idx) + start][need_idx],
                        keep_flag[start + len(need_idx) :],
                    ]
                )
            else:
                keep_flag = keep_flag[need_idx]
        
        need_idx_temp = need_idx[end - iterative_size:end]

        end -= (need_idx[:end] == 0).sum()
        return (
            compressed_input_ids,
            compressed_attention_mask,
            keep_flag,
            end,
            loss,
            self_loss,
            self_compressed_input_ids,
            self_compressed_attention_mask,
            # extract it
            need_idx_temp
        )

    def get_need_idx(
        self,
        loss,
        input_ids,
        attention_mask,
        end=200,
        iterative_size=200,
        threshold=0.5,
        keep_flag=None,
        split_token_id: int = 13,
        start: int = 0,
        self_loss=None,
        self_input_ids=None,
        self_attention_mask=None,
    ):
        if self_loss is not None:
            need_idx = torch.concat(
                [
                    loss[:start] > 0,
                    self_loss[: loss[start:].shape[0]] - loss[start:] > threshold,
                    loss[:1] > 0,
                ]
            )
        else:
            need_idx = torch.concat([loss > threshold, loss[:1] > 0])
        loss = loss[need_idx[:-1]]
        if self_loss is not None:
            if need_idx.shape[0] < self_loss.shape[0] + start + 1:
                need_idx = torch.cat(
                    [
                        need_idx,
                        torch.ones(
                            self_loss.shape[0] - need_idx.shape[0] + start + 1,
                            dtype=torch.bool,
                        ).to(need_idx.device),
                    ]
                )
            self_loss = self_loss[need_idx[start:-1]]

        if need_idx.shape[0] < input_ids.shape[1]:
            need_idx = torch.cat(
                [
                    need_idx,
                    torch.ones(
                        input_ids.shape[1] - need_idx.shape[0], dtype=torch.bool
                    ).to(need_idx.device),
                ]
            )
        elif need_idx.shape[0] > input_ids.shape[1]:
            need_idx = need_idx[: input_ids.shape[1]]

        return need_idx


# find specific index in tensor1 for all token ids in tensor2
def get_token_idx(tensor1, tensor2):
    index_list = []
    begin = 0
    for token_id in tensor2:
        idx_temp = 0
        for idx, original_id in enumerate(tensor1):
            if original_id == token_id:
                index_list.append(idx + begin)
                idx_temp = idx + 1
                begin += idx_temp
                break

        tensor1 = tensor1[idx_temp:]

    return index_list
