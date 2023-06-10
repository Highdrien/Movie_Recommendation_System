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

            You can choose a end_funtion which will be applied at each elements 
            of the rating matrix in order to have number between 1 and 5 
            if self.end_function='sigmoid', then the function f(x) = 4 * sigmoid(x) + 1
            if self.end_function='clamp', then the function f(x) = max( min(x, 1), 5)
            if self.end_function='tanh', then the function f(x) = 2 * tanh(x) + 3
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
        
        elif self.end_function == 'tanh':
            result = 2 * torch.tanh(result) + 3

        return result
    

# Définir le modèle de recommandation avec GCN
class MovieRecommendationModel_withMovieEmbedding(nn.Module):
    def __init__(self,
                 embedding_dim: int, 
                 hidden_layer_1: int, 
                 hidden_layer_2: int,
                 hidden_layer_3: int,
                 dropout: float,
                 end_function: bool,
                 max_users: int,
                 num_items: int) -> None:
        super(MovieRecommendationModel_withMovieEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_items + 1, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_layer_1)
        self.conv2 = GCNConv(hidden_layer_1, hidden_layer_2)
        self.linear = nn.Linear(hidden_layer_2, hidden_layer_3)
        self.linear_after_matmul = nn.Linear(1, 5)
        self.dropout = nn.Dropout(dropout)
        self.end_function = end_function
        self.num_items = num_items
        self.max_users = max_users
        if end_function == 'softmax':
            self.softmax = nn.Softmax(dim=2)

    def forward(self,
                x: torch.tensor,
                edge_index: torch.tensor,
                num_users: int) -> torch.tensor:
        users = torch.zeros(num_users).long()
        items = x[num_users:] - num_users + 1
        users_embedding = self.embedding(users)
        items_embedding = self.embedding(items)
        x = torch.concatenate((users_embedding, items_embedding), dim=0)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear(x)
        # x = F.relu(x)
        x = self.dropout(x)
        users = x[:num_users, :]
        items = x[num_users:, :]

        result = torch.matmul(users, items.t())

        if self.end_function == 'sigmoid':
            result = 4 * torch.sigmoid(result) + 1
        
        elif self.end_function == 'clamp':
            result = torch.clamp(result, min=1, max=5)
        
        elif self.end_function == 'tanh':
            result = 2 * torch.tanh(result) + 3
        
        elif self.end_function == 'softmax':
            result.requires_grad_(True)
            result = result.flatten()
            zeros_tensor = torch.zeros(((self.max_users - num_users) * self.num_items), requires_grad=True)
            result = torch.cat((result, zeros_tensor), dim=0)
            result = self.linear_after_matmul(result.unsqueeze(1))
            result = result[:num_users * self.num_items]
            index = torch.arange(num_users * self.num_items).reshape(num_users, self.num_items)
            result = result[index]
            result = self.softmax(result)
    
        return result


def get_model(config):
    if config.model.embedding_dim == 0:
        print('model without movies embedding')
        model = MovieRecommendationModel(hidden_layer_1=config.model.hidden_layer_1,
                                         hidden_layer_2=config.model.hidden_layer_2,
                                         hidden_layer_3=config.model.hidden_layer_3,
                                         dropout=config.model.dropout,
                                         end_function=config.model.end_function)
    else:
        print('model with movies embedding')
        model = MovieRecommendationModel_withMovieEmbedding(embedding_dim=config.model.embedding_dim,
                                                            hidden_layer_1=config.model.hidden_layer_1,
                                                            hidden_layer_2=config.model.hidden_layer_2,
                                                            hidden_layer_3=config.model.hidden_layer_3,
                                                            dropout=config.model.dropout,
                                                            end_function=config.model.end_function,
                                                            num_items=config.data.num_items,
                                                            max_users=600)
    # print(model)
    return model