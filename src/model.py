import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# Définir le modèle de recommandation avec GCN
class MovieRecommendationModel(nn.Module):
    def __init__(self, 
                 hidden_layer_1: int, 
                 hidden_layer_2: int,
                 hidden_layer_3: int,
                 dropout: float,
                 end_function: bool) -> None:
        super(MovieRecommendationModel, self).__init__()
        self.conv1 = GCNConv(1, hidden_layer_1)
        self.conv2 = GCNConv(hidden_layer_1, hidden_layer_2)
        self.linear = nn.Linear(hidden_layer_2, hidden_layer_3)
        self.dropout = nn.Dropout(dropout)
        self.end_function = end_function

    def forward(self,
                x: torch.tensor,
                edge_index: torch.tensor,
                num_users: int) -> torch.tensor:
        """
        Arg:
            - x is a concatened list of user_id and item_id. shape: (|V|)
            - edge_index is the tensor of the edges. shape: (2, |E|) 
            - num_users is the number of users
        
        Return:
            The predicted rating matrix.
            if self.end_function='sigmoid', then the function f(x)=4*sigmoid(x)+1 
                will be applied on all the element of the rating matrix
            if self.end_function='clamp', then the function f(x)=max(min(x, 1), 5)
                will be applied on all the element of the rating matrix
            otherwise, the element of rating matrix might not be between 1 and 5
        """
        x = x.unsqueeze(dim=-1).float()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)
        users = x[:num_users, :]
        items = x[num_users:, :]

        result = torch.matmul(users, items.t())

        if self.end_function == 'sigmoid':
            result = 4 * torch.sigmoid(result) + 1
        
        elif self.end_function == 'clamp':
            result = torch.clamp(result, min=1, max=5)

        return result


def get_model(config):
    return MovieRecommendationModel(hidden_layer_1=config.model.hidden_layer_1,
                                    hidden_layer_2=config.model.hidden_layer_2,
                                    hidden_layer_3=config.model.hidden_layer_3,
                                    dropout=config.model.dropout,
                                    end_function=config.model.end_function)