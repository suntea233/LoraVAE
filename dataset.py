import numpy as np
import torch
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class YelpDatabase(Dataset):
    def __init__(self,args):
        super(YelpDatabase, self).__init__()
        self.args = args
        self.path = self.args.path
        self.ckpt = self.args.ckpt
        self.train = self.args.train
        if self.train:
            self.path = self.path + r"\train.txt"
        else:
            self.path = self.path + r"\test.txt"

        self.input_ids = self.preprocessing()

        self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt, add_prefix_space=True)
        self.max_len = self.args.max_len



    def preprocessing(self):
        input_ids = []

        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split('\t')[1]
                input_ids.append(line)

        return input_ids


    def __len__(self):
        return len(self.input_ids)


    def __getitem__(self, item):
        return self.input_ids[item]

    def collate_fn(self,batch):
        input_ids = []
        for text in batch:
            input_ids.append([text])
        tokenized_inputs = self.tokenizer(input_ids, truncation=True,is_split_into_words=True,
                                           padding='max_length', max_length=self.max_len, return_tensors='pt')
        return tokenized_inputs['input_ids'],tokenized_inputs['attention_mask']