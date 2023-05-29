import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from src.model import get_model
from src.utils import save_learning_curves
from src.dataloader import creates_generators
from src.checkpoints import save_checkpoint_all, save_checkpoint_best, save_checkpoint_last

from config.utils import train_logger, train_step_logger


def train(config):

    train_loader, val_loader, _ = creates_generators(config)

    # Instancier le modèle
    model = get_model(config)

    # Définir la fonction de perte et l'optimiseur
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.model.learning_rate)

    logging_path = train_logger(config)
    best_epoch, best_val_loss = 0, 10e6

    ###############################################################
    # Start Training                                              #
    ###############################################################
    for epoch in range(config.train.epochs):
        model.train()
        train_loss = []
        
        train_range = tqdm(train_loader)
        for inputs, targets in train_range:
            optimizer.zero_grad()

            user_ids = inputs[:, 0]
            item_ids = inputs[:, 1]

            outputs = model(user_ids, item_ids)
            loss = criterion(outputs.squeeze(), targets.view(-1))
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            # train_loss += loss.item() * inputs.size(0)

            train_range.set_description("TRAIN -> epoch: %4d || loss: %4.4f" % (epoch, np.mean(train_loss)))
            train_range.refresh()

        train_loss = np.mean(train_loss)

        ###############################################################
        # Start Validation                                            #
        ###############################################################
        model.eval()
        val_loss = []
        val_range = tqdm(val_loader)

        with torch.no_grad():
            for inputs, targets in val_range:
                user_ids = inputs[:, 0]
                item_ids = inputs[:, 1]

                outputs = model(user_ids, item_ids)
                loss = criterion(outputs.squeeze(), targets)
                
                val_loss.append(loss.item())
                # val_loss += loss.item() * inputs.size(0)

                val_range.set_description("VAL   -> epoch: %4d || val_loss: %4.4f" % (epoch, np.mean(val_loss)))
                val_range.refresh()

        val_loss = np.mean(val_loss)

        ###################################################################
        # Save Scores in logs                                             #
        ###################################################################

        train_step_logger(logging_path, epoch, train_loss, val_loss, [], [])

        if config.model.save_checkpoint == 'all':
            save_checkpoint_all(model, logging_path, epoch)

        elif config.model.save_checkpoint == 'best':
            best_epoch, best_val_loss = save_checkpoint_best(model, logging_path, epoch, best_epoch, val_loss, best_val_loss)

    
    if config.model.save_checkpoint == 'best':
        save_checkpoint_best(model, logging_path, epoch, best_epoch, val_loss, best_val_loss, end_training=True)

    elif config.model.save_checkpoint == 'last':
        save_checkpoint_last(config, model, logging_path)

    if config.train.save_learning_curves:
        save_learning_curves(logging_path)
    
