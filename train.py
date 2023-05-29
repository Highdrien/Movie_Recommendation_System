import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from model import get_model
from dataloader import creates_generators

import parameters as PARAM

def train():

    train_loader, val_loader, test_loader = creates_generators()

    # Instancier le modèle
    model = get_model()

    # Définir la fonction de perte et l'optimiseur
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=PARAM.LR)

    # Entraînement du modèle
    for epoch in range(PARAM.NUM_EPOCHS):
        print('epoch:', epoch)
        model.train()
        train_loss = 0.0

        for inputs, targets in tqdm(train_loader):
            optimizer.zero_grad()

            user_ids = inputs[:, 0]
            item_ids = inputs[:, 1]

            outputs = model(user_ids, item_ids)
            loss = criterion(outputs.squeeze(), targets.view(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader)

        # Évaluation sur l'ensemble de validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader):
                user_ids = inputs[:, 0]
                item_ids = inputs[:, 1]

                outputs = model(user_ids, item_ids)
                loss = criterion(outputs.squeeze(), targets)

                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{PARAM.NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    torch.save(model.state_dict(), "model_with_dropout.pth")

    # Évaluation finale sur l'ensemble de test
    # model.eval()
    # test_loss = 0.0

    # with torch.no_grad():
    #     for inputs, targets in test_loader:
    #         user_ids = inputs[:, 0]
    #         item_ids = inputs[:, 1]

    #         outputs = model(user_ids, item_ids)
    #         loss = criterion(outputs.squeeze(), targets)

    #         test_loss += loss.item() * inputs.size(0)

    # test_loss /= len(test_loader)

    # print(f"Test Loss: {test_loss:.4f}")


train()

