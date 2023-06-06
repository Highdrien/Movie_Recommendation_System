import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, target, predict):
        mask = (predict != 0).float()       # Créer un masque binaire où 0 devient 0 et tout autre nombre devient 1
        masked_predict = predict * mask     # Appliquer le masque à la prédiction
        masked_target = target * mask       # Appliquer le masque à la cible
        loss = self.mse_loss(masked_predict, masked_target)
        loss = torch.mean(loss)             # Prendre la moyenne de la loss sur tous les éléments
        return loss
