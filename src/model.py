import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


# Définir le modèle de recommandation avec GCN
class MovieRecommendationModel(nn.Module):
    def __init__(self, num_users, num_items):
        super(MovieRecommendationModel, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, 32)
        self.item_embeddings = nn.Embedding(num_items, 32)
        self.conv1 = GCNConv(32, 64)
        self.conv2 = GCNConv(64, 2)
        self.num_users = num_users

    def forward(self, user_ids, item_ids, edge_index):
        user_embeds = self.user_embeddings(user_ids)
        item_embeds = self.item_embeddings(item_ids)
        x = torch.cat((user_embeds, item_embeds), dim=0)
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        users = x[:self.num_users, :]
        items = x[self.num_users:, :]

        result = torch.matmul(users, items.t())

        # Avoid numbers less that 1 or more that 5
        result = torch.clamp(result, min=1, max=5)

        return result


def get_model(config):
    return MovieRecommendationModel(num_users=config.data.nb_users,
                                    num_items=config.data.nb_items)