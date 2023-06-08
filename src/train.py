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
    train_data = get_data(config, 'train')
    val_data = get_data(config, 'val')

    train_num_users = train_data.get_num_user()
    val_num_users = val_data.get_num_user()

    train_ids, train_edge_index = train_data.get_input()
    val_ids, val_edge_index = val_data.get_input()

    train_target = train_data.get_target()
    val_target = val_data.get_target()

    # Get model
    model = get_model(config)

    # Loss and Optimizer
    criterion = MaskedMSELoss()
    # criterion = AdvancedMaskedMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.model.learning_rate)

    if config.train.logs:
        logging_path = train_logger(config)
    best_epoch, best_val_loss = 0, 10e6

    train_loss_list = []
    val_loss_list = []

    epochs_range = tqdm(list(range(1, config.train.epochs + 1)))
    for epoch in epochs_range:
        current_best = False

        # Training
        model.train()
        train_predict = model(train_ids, train_edge_index, train_num_users)
        loss = criterion(target=train_target, predict=train_predict)
        train_loss = loss.item()
        train_loss_list.append(train_loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Validation
        with torch.no_grad():
            model.eval()
            val_predict = model(val_ids, val_edge_index, val_num_users)
            loss = criterion(target=val_target, predict=val_predict)
            val_loss = loss.item()
            val_loss_list.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                current_best = True

        epochs_range.set_description("epoch: %4d || loss: %4.4f || val_loss: %4.4f" % (epoch, train_loss, val_loss))
        epochs_range.refresh()

        # Save Scores in logs
        if config.train.logs:
            train_step_logger(logging_path, epoch, train_loss, val_loss, [], [])

            # Save model according the configuration
            if config.model.save_checkpoint == 'all':
                save_checkpoint_all(model, logging_path, epoch)

            elif config.model.save_checkpoint == 'best' and current_best:
                save_checkpoint_best(model, logging_path, epoch)

    # Save Scores in logs at the end of training
    if config.train.logs:
        if config.model.save_checkpoint == 'best':
            save_checkpoint_best(model, logging_path, best_epoch, end_training=True)

        elif config.model.save_checkpoint == 'last':
            save_checkpoint_last(config, model, logging_path)

        if config.train.save_learning_curves:
            save_learning_curves(logging_path)

    print('best val loss:', best_val_loss, 'in the epoch:', best_epoch)

    if config.train.logs:
        return logging_path