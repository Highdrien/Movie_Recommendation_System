import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        # self.clamp = clamp

    def forward(self, target: torch.tensor, predict: torch.tensor):
        # if self.clamp:
        #     # Avoid numbers less that 1 or more that 5
        #     predict = torch.clamp(predict, min=1, max=5)
        
        target = target.flatten().float()
        predict = predict.flatten().float()
        mask = (predict != 0).float()       # Créer un masque binaire où 0 devient 0 et tout autre nombre devient 1
        masked_predict = predict * mask  # Appliquer le masque à la prédiction
        masked_target = target * mask       # Appliquer le masque à la cible
        loss = self.mse_loss(masked_predict, masked_target)
        loss = torch.mean(loss)             # Prendre la moyenne de la loss sur tous les éléments
        return loss


# class AdvancedMaskedMSELoss(nn.Module):
#     def __init__(self):
#         super(AdvancedMaskedMSELoss, self).__init__()
#         self.mse_loss = nn.MSELoss(reduction='none')

#     def forward(self, target, predict):
#         target = target.flatten().float()
#         predict = predict.flatten().float()

#         mask = (target != 0).float()  # Créer un masque binaire où 0 devient 0 et tout autre nombre devient 1
        
#         # Calculer la loss pour les cases où target != 0
#         non_zero_mask = mask.clone()
#         loss_non_zero = self.mse_loss(predict, target) * non_zero_mask
        
#         # Appliquer la logique avancée pour les cases où target = 0
#         zero_mask = (target == 0).float()
#         less_than_1_mask = (predict < 1).float()
#         greater_than_5_mask = (predict > 5).float()
        
#         # Calculer la loss lorsque predict < 1 et target = 0
#         loss_less_than_1 = self.mse_loss(predict, torch.ones_like(predict)) * zero_mask * less_than_1_mask
        
#         # Calculer la loss lorsque predict > 5 et target = 0
#         loss_greater_than_5 = self.mse_loss(predict, torch.ones_like(predict) * 5) * zero_mask * greater_than_5_mask
        
#         # Combinaison des différentes losses
#         loss = torch.mean(loss_non_zero) + torch.mean(loss_less_than_1) + torch.mean(loss_greater_than_5)
        
#         return loss
