import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

'''
todo: implement it works for all os by import os
dataset: sample & label
dataloder: make dataset iterable
'''

def ML1m():
    df = pd.read_csv(
            "../ml-1m/ratings.dat",
            sep="::",
            names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
            engine='python')
    df = df.drop(columns=['Timestamp'])

    # Sample Train and Test Dataset in Random
    training_dataset = df.sample(frac=0.9, random_state=200)
    test_dataset = df.drop(train.index).sample(frac=1.0)

    training_dataset=training_dataset.reset_index(drop=True)
    test_dataset=test_dataset.reset_index(drop=True)
    return training_dataset, test_dataset

class MFDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user = self.df.iloc[idx, 0]
        item = self.df.iloc[idx, 1]
        rating = self.df.iloc[idx, 2]
        return {"user": user, "item": item, "rating": rating}

'''
Real Data
training_data, test_data = ML1m()
trainDataSet=MFDataset(training_data)
train_dataloader = DataLoader(trainDataSet, batch_size=65, shuffle=True)
batch_iterator = iter(train_dataloader)
data = next(batch_iterator)

'''

data = [[0, 1, 3], [0, 2, 1], [1, 3, 2], [1, 1, 5], [2, 1, 3], [2, 2, 3], [3, 1, 5]]

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class MF(torch.nn.Module):
    def __init__(self, global_mean, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_embeddings = torch.nn.Embedding(n_users, n_factors)
        self.item_embeddings = torch.nn.Embedding(n_items, n_factors)
        self.user_biases = torch.nn.Embedding(n_users, 1)
        self.item_biases = torch.nn.Embedding(n_items, 1)
        self.global_mean =  global_mean

    def forward(self, user, item):
        preds = self.user_biases(user) + self.item_biases(item)
        preds = preds + (self.user_embeddings(user) * self.item_embeddings(item)).sum(1)
        preds = preds + self.global_mean
        return preds

global_mean = 0
for e in data:
    global_mean = global_mean + e[2]
global_mean = global_mean / len(data)
model = MF(global_mean, 4, 4, 10)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

dataset = CustomDataset(data)
train_loader = DataLoader(dataset, batch_size=5,shuffle=True)

for epoch in range(30):
    for _, batch in enumerate(train_loader):
        u, i, r = batch
        r = r.float()

        preds = model(u, i)
        loss = criterion(preds, r)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/30], Loss:{loss.item()}')
