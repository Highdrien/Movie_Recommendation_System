import numpy as np

import torch

from src.model import get_model
from src.loss import MaskedMSELoss
from src.dataloader import get_data
from src.checkpoints import get_checkpoint_path

from config.utils import test_logger

def test(logging_path, config):

    # get data
    user_ids, item_ids, edge_index, target = get_data(config)

    # Split target for train and validation
    num_users = target.shape[0]
    split_2 = int(num_users * config.data.split_2) + int(num_users * config.data.split_1)
    test_target = target[split_2:]

    del target

    model = get_model(config)

    # Load model's weight
    checkpoint_path = get_checkpoint_path(config, logging_path)
    model.load_state_dict(torch.load(checkpoint_path))

    # Définir la loss
    criterion = MaskedMSELoss()

    # Évaluation finale sur l'ensemble de test
    model.eval()

    with torch.no_grad():
        model.eval()
        test_predict = model(user_ids, item_ids, edge_index)[split_2:]
        loss = criterion(target=test_target, predict=test_predict)
        test_loss = loss.item()

    print('test loss:', test_loss)
    test_logger(logging_path, ['MaskedMSELoss'], [test_loss])

