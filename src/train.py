from tqdm import tqdm

import torch
import torch.optim as optim

from src.model import get_model
from src.loss import MaskedMSELoss
from src.dataloader import get_data
from src.utils import save_learning_curves
from src.checkpoints import save_checkpoint_all, save_checkpoint_best, save_checkpoint_last

from config.utils import train_logger, train_step_logger


def train(config):

    # get data
    user_ids, item_ids, edge_index, target = get_data(config)

    # Split target for train and validation
    num_users = target.shape[0]
    split_1 = int(num_users * config.data.split_1)
    split_2 = int(num_users * config.data.split_2) + split_1
    train_target = target[:split_1]
    val_target = target[split_1:split_2]

    del target

    model = get_model(config)

    # Définir la fonction de perte et l'optimiseur
    criterion = MaskedMSELoss()
    # criterion = AdvancedMaskedMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.model.learning_rate)

    logging_path = train_logger(config)
    best_epoch, best_val_loss = 0, 10e6

    train_loss_list = []
    val_loss_list = []

    epochs_range = tqdm(list(range(1, config.train.epochs + 1)))
    for epoch in epochs_range:
        current_best = False

        # Training
        model.train()
        train_predict = model(user_ids, item_ids, edge_index)[:split_1]
        loss = criterion(target=train_target, predict=train_predict)
        train_loss = loss.item()
        train_loss_list.append(train_loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Validation
        with torch.no_grad():
            model.eval()
            val_predict = model(user_ids, item_ids, edge_index)[split_1:split_2]
            loss = criterion(target=val_target, predict=val_predict)
            val_loss = loss.item()
            val_loss_list.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                current_best = False

        epochs_range.set_description("epoch: %4d || loss: %4.4f || val_loss: %4.4f" % (epoch, train_loss, val_loss))
        epochs_range.refresh()

        # Save Scores in logs
        train_step_logger(logging_path, epoch, train_loss, val_loss, [], [])

        # Save model according the configuration
        if config.model.save_checkpoint == 'all':
            save_checkpoint_all(model, logging_path, epoch)

        elif config.model.save_checkpoint == 'best' and current_best:
            save_checkpoint_best(model, logging_path, epoch, val_loss)

    
    if config.model.save_checkpoint == 'best':
        save_checkpoint_best(model, logging_path, epoch, best_epoch, val_loss, best_val_loss, end_training=True)

    elif config.model.save_checkpoint == 'last':
        save_checkpoint_last(config, model, logging_path)

    if config.train.save_learning_curves:
        save_learning_curves(logging_path)

    print('best val loss:', best_val_loss, 'in the epoch:', best_epoch)

    return logging_path