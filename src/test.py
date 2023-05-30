from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn

from src.model import get_model
from src.dataloader import creates_generators
from src.checkpoints import get_checkpoint_path

from config.utils import test_logger

def test(logging_path, config):

    _, _, test_loader = creates_generators(config)

    # Instancier le modèle
    model = get_model(config)

    # Load model's weight
    checkpoint_path = get_checkpoint_path(config, logging_path)
    model.load_state_dict(torch.load(checkpoint_path))

    # Définir la loss
    criterion = torch.nn.CrossEntropyLoss()

    # Évaluation finale sur l'ensemble de test
    model.eval()
    test_loss = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            user_ids = inputs[:, 0]
            item_ids = inputs[:, 1]
            
            outputs = model(user_ids, item_ids)
            loss = criterion(outputs.squeeze(), targets)

            test_loss.append(loss.item())
            

    test_loss = np.mean(test_loss)

    test_logger(logging_path, ['MSE'], [test_loss])

