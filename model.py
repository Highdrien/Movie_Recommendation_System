import torch
import torch.nn as nn

# Définir le modèle de recommandation
class MovieRecommendationModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MovieRecommendationModel, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embeddings(user_ids)
        item_embeds = self.item_embeddings(item_ids)
        x = torch.cat((user_embeds, item_embeds), dim=1)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x


def get_model(num_users, num_items, embedding_dim):
    return MovieRecommendationModel(num_users, num_items, embedding_dim)