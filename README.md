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
python train3hop-cls-segment.py

Note: It will load the file 'all_kg_new.py'
Prepare for directories: The data file contains a subfolder must in "hop1 or hop2 or hop3", the subfiles are MetaQA data for each hop.
```

## Acknowledgement
1. [EmbedKGQA](https://github.com/malllabiisc/EmbedKGQA)
