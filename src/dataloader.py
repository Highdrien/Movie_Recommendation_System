import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


torch.random.seed()


# Dataset class
class MovieDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tensor = self.inputs[idx]
        target_tensor = self.targets[idx]
        return input_tensor, target_tensor


def creates_generators(config):
    # Load data
    data = pd.read_csv(config.data.path)

    item_encoder = LabelEncoder()
    data['item_id'] = item_encoder.fit_transform(data['item_id'])

    # split data: (user_id, item_id) and target (rating)
    inputs = data[['user_id', 'item_id']].values
    targets = data['rating'].values

    # split data: train, val and test
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, test_size=0.2, random_state=42)
    train_inputs, val_inputs, train_targets, val_targets = train_test_split(train_inputs, train_targets, test_size=0.2, random_state=42)

    # Create generators
    train_dataset = MovieDataset(train_inputs, train_targets)
    val_dataset = MovieDataset(val_inputs, val_targets)
    test_dataset = MovieDataset(test_inputs, test_targets)

    # use dataloader
    train_loader = DataLoader(train_dataset, 
                            batch_size=config.train.batch_size, 
                            shuffle=config.train.shuffle,
                            drop_last=config.train.drop_last)
    
    val_loader = DataLoader(val_dataset, 
                            batch_size=config.val.batch_size, 
                            shuffle=config.val.shuffle,
                            drop_last=config.val.drop_last)
    
    test_loader = DataLoader(test_dataset, 
                            batch_size=config.test.batch_size, 
                            shuffle=config.test.shuffle,
                            drop_last=config.test.drop_last)

    return train_loader, val_loader, test_loader
