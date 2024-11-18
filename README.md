# <div align="center">Lighter And Better: Towards Flexible Context Adaptation For Retrieval Augmented Generation<div>

<div align="center">
<a href="https://arxiv.org/abs/2409.15699" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/wcyno23/FlexRAG" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Model-27b3b4.svg></a>
<a href="https://github.com/"><img alt="License" src="https://img.shields.io/badge/Apache-2.0-green"></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>
<h4 align="center">

## 👋 Overview

**FlexRAG** is a lightweight model designed to reduce RAG running costs while improving its generation quality. It compresses the retrieved contexts into compact embeddings and these embedding are optimized to enhance downstream RAG performance. A key feature of FlexRAG is its flexibility, which enables effective support for diverse compression ratios and selective preservation of important contexts. 

## 🛠️ Set up

### Data

The evaluation dataset for FlexRAG is released [here](https://huggingface.co/datasets/wcyno23/FlexRAG-eval). Please download and unzip them to the `data` folder.

### Environment

You can install the necessary dependencies using the following command. Recommended Python version is 3.10+.

```
pip install -r requirements.txt
```

## :rocket: Usage

The entire experiment scripts are included at the `experiments` directory. For example, to evaluate on Long-sequence Multi-doc QA dataset:

Vanilla RAG:

```
cd FlexRAG
bash experiments/eval/eval_longbench_base.sh
```

FlexRAG w/o Selective Compression:

```
bash experiments/eval/eval_longbench_flexrag_wo_sc.sh
```

FlexRAG w. Selective Compression:

```
bash experiments/eval/eval_longbench_flexrag_embedding.sh
```

The final evaluation results will be stored in the `data/longbench` directory.


## ✍️ Citation
If you find this repository useful, please consider giving a star ⭐ and citation
```
@article{liu2024lighter,
  title={Lighter And Better: Towards Flexible Context Adaptation For Retrieval Augmented Generation},
  author={Liu, Zheng and Wu, Chenyuan and Shao, Ninglu and Xiao, Shitao and Li, Chaozhuo and Lian, Defu},
  journal={arXiv preprint arXiv:2409.15699},
  year={2024}
}
```