import torch
from torch.utils.data import random_split, DataLoader

def train_data_loader(data_train, batchsize):
    train_loader=DataLoader(dataset=data_train, 
                            batch_size=batchsize,
                            shuffle=True, num_workers=2)
    return train_loader