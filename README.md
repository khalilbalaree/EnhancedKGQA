# EnhancedKGQA

## Introduction
1. An CMPUT 651 course project: An improved end-to-end method to knowledge graph embedding based question answering.  
2. Based on ACL2020 paper: Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings.

## Instructions

In order to run the code, first download data.zip and pretrained_model.zip from https://drive.google.com/drive/folders/1RlqGBMo45lTmWz9MUPTq-0KcjSd3ujxc?usp=sharing. Unzip these files in the main directory.

## Excution for 1 & 2 hop dataset

```bash
pip install transformers
pip install pytorch-lightning

python train.py
```
## Excution for 3 hop dataset

```
python train
```

## Acknowledgement
1. [EmbedKGQA](https://github.com/malllabiisc/EmbedKGQA)
