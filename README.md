# EnhancedKGQA

## Introduction
1. An CMPUT 651 course project: An improved end-to-end method to knowledge graph embedding based question answering.  
2. Based on ACL2020 paper: Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings.

## Instructions

In order to run the code, first download data.zip and pretrained_model.zip from https://drive.google.com/drive/folders/1RlqGBMo45lTmWz9MUPTq-0KcjSd3ujxc?usp=sharing. Unzip these files in the main directory.

The data.zip has the data files that in "EmbededKGQA/data/QA_data/MetaQa/", which need to be put into /data subfolder

The pretrained_model.zip has the pre-trained model files that in "EmbedKGQA/pretrained_models/embeddings/ComplEx_MetaQA_full/", which need to be put into ./EnhancedKGQA_main/ComplEx_MetaQA_full/
## Excution for 1 & 2 hop dataset
Prepare for directories: 

(1) The data file contains a subfolder must in "hop1 or hop2 or hop3", the subfiles are MetaQA data for each hop (such as "qa_train_3hop.txt").

(2) In all_kg_new.py file, the default subfolder for load pre-trained kg models are "./EnhancedKGQA_main/ComplEx_MetaQA_full/" and "./EnhancedKGQA_main/data/MetaQA/train.txt"

(3) In all_kg_new.py file, the default subfolder for load pre-trained Complex is "./EnhancedKGQA_main/ComplEx_MetaQA_full/"
```bash
pip install transformers
pip install pytorch-lightning

python train.py
```
## Excution for 3 hop dataset

Prepare for directories: 

(1) The data file contains a subfolder must in "hop1 or hop2 or hop3", the subfiles are MetaQA data for each hop (such as "qa_train_3hop.txt").

(2) In all_kg_new.py file, the default subfolder for load pre-trained kg models are "./EnhancedKGQA_main/ComplEx_MetaQA_full/" and "./EnhancedKGQA_main/data/MetaQA/kb.txt"

(3) In all_kg_new.py file, the default subfolder for load pre-trained Complex is "./EnhancedKGQA_main/ComplEx_MetaQA_full/"
```
python train3hop-cls-segment.py

Note: It will load the file 'all_kg_new.py'
```

## Acknowledgement
1. [EmbedKGQA](https://github.com/malllabiisc/EmbedKGQA)
