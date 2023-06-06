import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# Définir le modèle de recommandation avec GCN
class MovieRecommendationModel(nn.Module):
    def __init__(self, 
                 num_users: int, 
                 num_items: int, 
                 embedding_dim: int, 
                 hidden_layer_1: int, 
                 hidden_layer_2: int,
                 dropout: float):
        super(MovieRecommendationModel, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_layer_1)
        self.conv2 = GCNConv(hidden_layer_1, hidden_layer_2)
        self.dropout = nn.Dropout(dropout)
        self.num_users = num_users

    def forward(self, user_ids, item_ids, edge_index):
        user_embeds = self.user_embeddings(user_ids)
        item_embeds = self.item_embeddings(item_ids)
        x = torch.cat((user_embeds, item_embeds), dim=0)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        users = x[:self.num_users, :]
        items = x[self.num_users:, :]

        result = torch.matmul(users, items.t())
        result = 4 * torch.sigmoid(result) + 1

        return result


def get_model(config):
    return MovieRecommendationModel(num_users=config.data.num_users,
                                    num_items=config.data.num_items,
                                    embedding_dim=config.model.embedding_dim,
                                    hidden_layer_1=config.model.hidden_layer_1,
                                    hidden_layer_2=config.model.hidden_layer_2,
                                    dropout=config.model.dropout)