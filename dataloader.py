import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import parameters as PARAM

torch.random.seed()


# Définir une classe de Dataset personnalisée
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


def creates_generators():
    # Load data
    data = pd.read_csv(PARAM.DATA_PATH)

    item_encoder = LabelEncoder()
    data['item_id'] = item_encoder.fit_transform(data['item_id'])

    # Séparer les données en entrées (user_id, item_id) et cibles (rating)
    inputs = data[['user_id', 'item_id']].values
    targets = data['rating'].values

    # Diviser les données en ensembles d'entraînement, de validation et de test
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, test_size=0.2, random_state=42)
    train_inputs, val_inputs, train_targets, val_targets = train_test_split(train_inputs, train_targets, test_size=0.2, random_state=42)

    # Créer des tenseurs PyTorch pour les ensembles d'entraînement, de validation et de test
    train_dataset = MovieDataset(train_inputs, train_targets)
    val_dataset = MovieDataset(val_inputs, val_targets)
    test_dataset = MovieDataset(test_inputs, test_targets)

    # Créer des DataLoaders pour faciliter l'itération sur les données
    train_loader = DataLoader(train_dataset, batch_size=PARAM.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=PARAM.BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=PARAM.BATCH_SIZE)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    creates_generators()