import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import auroc
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import re
import os


def extract_questions_and_answers(factoid_path: Path):
    if "hop1" in str(factoid_path):
        #print(factoid_path)
        type_id = 0
    elif "hop2" in str(factoid_path):
        type_id = 1
    elif "hop3" in  str(factoid_path):
        type_id = 2
    data = open(factoid_path, "r+",encoding='utf-8')
    Lines = data.readlines()
    data_rows = []
    counter = 0
    for line in Lines:
        question, ans = line.split("\t")
        sep = re.split("\[(.*?)\]", question)
        entity = re.findall("\[(.*?)\]", question)
        new_question = "".join(sep)
        assert len(entity) == 1
        assert len(sep) == 3
        # print(question)

        # print(entity[0])
        # print(sep)
        # print(new_question)
        question_length = len(new_question.split(" "))
        start_idx = len(sep[0].split(" ")) - 1
        end_idx = len(sep[1].split(" ")) + start_idx - 1
        # print(question_length,start_idx,end_idx)
        ans = ans.strip("\n").split('|')


        data_rows.append({
            "question": new_question,
            "entity": entity[0],
            "answer": ans,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "sentence_length": question_length,
            "hop_type":type_id
        })
        counter += 1
    return pd.DataFrame(data_rows)
def prepare_data():
    RANDOM_SEED = 42
    sns.set(style='whitegrid', palette='muted', font_scale=1.2)
    HAPPY_COLORS_PALETTE = ["#01BEFE", '#FFDD00', '#FF7D00', '#FF006D', '#ADFF02', '#8F00FF']
    sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
    rcParams['figure.figsize'] = 12, 8

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


    factoid_paths = []
    for folder in os.listdir("./data"):
        filename = "./data/"+folder
        factoid_paths+=list(Path(filename + "/").glob("qa_*.txt"))
    print(factoid_paths)
    #print(1+'1')
    dfs = []
    for factoid_path in factoid_paths:
        a_df = extract_questions_and_answers(factoid_path)
        dfs.append(a_df)
    df = pd.concat(dfs)
    return df




class Dataset(Dataset):
    def __init__(self, entity_dict : dict,data: pd.DataFrame, tokenizer: BertTokenizer, max_token_len: int = 64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.entity_dict = entity_dict

    def __len__(self):
        return len(self.data)

    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        batch_size = len(indices)
        vec_len = len(self.entity_dict)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        # one_hot = -torch.ones(vec_len, dtype=torch.float32)
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        new_question = data_row.question
        ans = data_row.answer
        #TODO 预处理 答案 变成 vector
        tail_ids = []
        for each_ans in ans:
            #print(each_ans)
            id = self.entity_dict[each_ans]
            #print(id)
            tail_ids.append(id)
        tail_onehot = self.toOneHot(tail_ids).to("cuda:0")
        #print(tail_onehot.shape)



        start_idx = data_row.start_idx
        end_idx = data_row.end_idx
        hop_type = data_row.hop_type
        encoding_question = self.tokenizer.encode_plus(
            new_question,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        return dict(
            question=new_question,
            answer = tail_onehot,
            input_ids=encoding_question["input_ids"].flatten(),
            attention_mask=encoding_question["attention_mask"].flatten(),
            start_idx_label=torch.tensor(start_idx),
            end_idx_label = torch.tensor(end_idx),
            hop_type = torch.tensor(hop_type)
        )



class DataModule(pl.LightningDataModule):

  def __init__(self,entity_dict, train_df, test_df, tokenizer,batchsize = 1,max_token_len = 64):
    super().__init__()
    self.train_df = train_df
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.batchsize = batchsize
    self.max_token_len = max_token_len
    self.entity_dict = entity_dict
  def setup(self):
    self.train_dataset = Dataset(
        self.entity_dict,
        self.train_df,
        self.tokenizer,
        self.max_token_len

    )
    self.test_dataset = Dataset(
        self.entity_dict,
        self.test_df,
        self.tokenizer,
        self.max_token_len
    )
  def train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        batch_size = self.batchsize,
        shuffle = True
    )
  def val_dataloader(self):
    return DataLoader(
        self.test_dataset,
        batch_size = self.batchsize
    )