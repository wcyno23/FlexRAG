# Evaluation

FlexRAG is evaluated on two types of datasets: Long-sequence Multi-document QA(LMQA) and Open-Domain QA(ODQA).

- LMQA: HotpotQA, 2WikiMQA and Musique
- ODQA:  NQ, PopQA and TriviaQA

The entire experiment scripts are included at the `experiments` directory.

## LMQA

Evaluate on Long-sequence Multi-doc QA dataset:

**Vanilla RAG:**

```bash
bash experiments/eval/eval_longbench_rag.sh
```

**FlexRAG w/o Selective Compression:**

```bash
bash experiments/eval/eval_longbench_flexrag_uniform.sh
```

**FlexRAG w. Selective Compression:**

By default, we employ sentence-level selective compression, which yields the most optimal results.

```bash
bash experiments/eval/eval_longbench_flexrag_sentence_level_sc.sh
```

The repository also supports token-level estimation through [LLMLingua](https://github.com/microsoft/LLMLingua).

```bash
bash experiments/eval/eval_longbench_flexrag_token_level_sc.sh
```


The final evaluation results will be stored in the `data/longbench` directory.

## ODQA

Evaluate on Open-Domain QA dataset:

**Vanilla RAG:**

```bash
bash experiments/eval/eval_open_domain_qa_rag.sh
```

**FlexRAG w/o Selective Compression:**

```bash
bash experiments/eval/eval_open_domain_qa_flexrag_uniform.sh
```

**FlexRAG w. Selective Compression:**

By default, we employ sentence-level selective compression, which yields the most optimal results.

```bash
bash experiments/eval/eval_open_domain_qa_flexrag_sentence_level_sc.sh
```

The repository also supports token-level estimation through [LLMLingua](https://github.com/microsoft/LLMLingua).

```bash
bash experiments/eval/eval_open_domain_qa_flexrag_sentence_level_sc.sh
```

The final evaluation results will be stored in the `data/open_domain_qa` directory.