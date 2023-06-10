import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, target: torch.tensor, predict: torch.tensor):  
        target = target.flatten().float()
        predict = predict.flatten().float()
        mask = (target != 0).float()            # Créer un masque binaire où 0 devient 0 et tout autre nombre devient 1
        masked_predict = predict * mask         # Appliquer le masque à la prédiction
        masked_target = target * mask           # Appliquer le masque à la cible
        loss = self.mse_loss(masked_predict, masked_target)
        return loss


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.num_classes = 5

    def forward(self, target: torch.tensor, predict: torch.tensor):
        mask = (target != 0).float()  # Créer un masque binaire où 0 devient 0 et tout autre nombre devient 1

        # Convertir la cible en vecteurs one-hot
        one_hot_target = F.one_hot(target.long(), num_classes=self.num_classes + 1).float()
        one_hot_target = one_hot_target[:, :, 1:]
        
        masked_predict = predict * mask.unsqueeze(2)  # Appliquer le masque à la prédiction
        masked_target = one_hot_target * mask.unsqueeze(2)  # Appliquer le masque à la cible
        
        masked_predict = masked_predict.view(-1, self.num_classes)  # Aplatir la prédiction
        masked_target = masked_target.view(-1, self.num_classes)  # Aplatir la cible
        
        loss = self.cross_entropy_loss(masked_predict, masked_target)
        loss = torch.mean(loss)
       
        return loss
