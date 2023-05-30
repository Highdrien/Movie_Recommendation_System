import torch
import torch.nn as nn


# Définir le modèle de recommandation
class MovieRecommendationModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, dropout):
        super(MovieRecommendationModel, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 5)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embeddings(user_ids)
        item_embeds = self.item_embeddings(item_ids)
        x = torch.cat((user_embeds, item_embeds), dim=1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.softmax(x)
        return x


def get_model(config):
    return MovieRecommendationModel(num_users=config.data.nb_users,
                                    num_items=config.data.nb_items,
                                    embedding_dim=config.model.embedding_dim,
                                    dropout=config.model.dropout)