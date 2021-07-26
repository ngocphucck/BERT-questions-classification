# Question classification task using BERT

## Introduction
BERT stands for Bidirectional Encoder Representations from Transformers - which is a transformer-based machine learning technique for natural language processing (NLP) pre-training developed by Google. BERT is produced through 2 approches: 
- Pretrained with Unsupervised Feature-based Approaches which consists of 2 task Mask LM and Next sentence prediction in huge datasets (the BooksCorpus with 800M wordsand English Wikipedia with 2,500M words)
- Fine-tuning BERT

BERT’s model architecture is a multi-layer bidirectional Transformer's encoder which is based on the original paper [Attention is all you need](https://arxiv.org/abs/1706.03762v5).
## Requirements
- python3
- pytorch
- transformers library
## Usage
```bash
$ python3 train.py
```
## Citation
```bibtex
@misc{devlin2019bert,
      title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding}, 
      author={Jacob Devlin and Ming-Wei Chang and Kenton Lee and Kristina Toutanova},
      year={2019},
      eprint={1810.04805},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
