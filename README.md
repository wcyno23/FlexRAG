# <div align="center">Lighter And Better: Towards Flexible Context Adaptation For Retrieval Augmented Generation<div>

<div align="center">
<a href="https://arxiv.org/abs/2409.15699" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://dl.acm.org/doi/abs/10.1145/3701551.3703580" target="_blank"><img src="https://img.shields.io/badge/ACM%20DL-Paper-blue?logo=acm"></a>
<a href="https://huggingface.co/wcyno23/FlexRAG" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Model-27b3b4.svg></a>
<a href="https://huggingface.co/datasets/wcyno23/TacZip-Data" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Dataset-ff69b4.svg"></a>
<a href="https://github.com/"><img alt="License" src="https://img.shields.io/badge/Apache-2.0-green"></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>
<h4 align="center">

## 👋 Overview

**FlexRAG** is a lightweight model designed to reduce RAG running costs while improving its generation quality. It compresses the retrieved contexts into compact embeddings and these embeddings are optimized to enhance downstream RAG performance. A key feature of FlexRAG is its flexibility, which enables effective support for diverse compression ratios and selective preservation of important contexts. 

![](imgs/frame.png)

## 🛠️ Set up

### Data

The evaluation dataset for FlexRAG is released [here](https://huggingface.co/datasets/wcyno23/FlexRAG-eval). Please download and unzip them to the `data` folder.

### Environment

You can install the necessary dependencies using the following command. Recommended Python version is 3.10+.

```
pip install -r requirements.txt
```

## :rocket: Usage

### Evaluation

The entire experiment scripts are included at the `experiments` directory. For example, to evaluate on Long-sequence Multi-doc QA dataset:

Vanilla RAG:

```
cd FlexRAG
bash experiments/eval/eval_longbench_rag.sh
```

FlexRAG w/o Selective Compression:

```
bash experiments/eval/eval_longbench_flexrag_uniform.sh
```

FlexRAG w. Selective Compression:

```
bash experiments/eval/eval_longbench_flexrag_sentence_level_sc.sh
```

The final evaluation results will be stored in the `data/longbench` directory.

### Training

See [training section](./examples/training.md).

## ✍️ Citation
If you find this repository useful, please consider giving a star ⭐ and citation
```
@inproceedings{wu2025lighter,
  title={Lighter and better: Towards flexible context adaptation for retrieval augmented generation},
  author={Wu, Chenyuan and Shao, Ninglu and Liu, Zheng and Xiao, Shitao and Li, Chaozhuo and Zhang, Chen and Wang, Senzhang and Lian, Defu},
  booktitle={Proceedings of the Eighteenth ACM International Conference on Web Search and Data Mining},
  pages={271--280},
  year={2025}
}
```