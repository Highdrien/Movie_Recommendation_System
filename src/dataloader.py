import numpy as np
import pandas as pd
import torch


def get_data(config):
    # Load data
    data = pd.read_csv(config.data.path)

    num_users = config.data.nb_users
    num_items = config.data.nb_items

    # Get edge_index
    user_ids = torch.tensor(data['user_id'].values, dtype=torch.long)
    item_ids = torch.tensor(data['item_id'].values, dtype=torch.long) - 1
    edge_index = torch.stack([user_ids, item_ids + num_users], dim=0)

    # Get rating matrix
    rating_matrix = np.zeros((num_users, num_items))
    for _, row in data.iterrows():
        user_id = row['user_id']
        item_id = row['item_id'] - 1
        rating = row['rating']
        rating_matrix[user_id, item_id] = rating
    
    target = torch.tensor(rating_matrix)

    user_ids = torch.arange(num_users).long()
    item_ids = torch.arange(num_items).long()

    return user_ids, item_ids, edge_index, target
