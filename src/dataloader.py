import os
import numpy as np
import pandas as pd
import torch


class Data():
    def __init__(self, data_path, num_items) -> None:
        # Load data
        data = pd.read_csv(data_path)

        self.num_users = len(data['user_id'].unique())
        self.num_items = num_items

        # Get edge_index
        user_ids = torch.tensor(data['user_id'].values, dtype=torch.long)
        item_ids = torch.tensor(data['item_id'].values, dtype=torch.long) - 1
        self.edge_index = torch.stack([user_ids, item_ids + self.num_users], dim=0)

        # Get rating matrix
        rating_matrix = np.zeros((self.num_users, self.num_items))
        for _, row in data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id'] - 1
            rating = row['rating']
            rating_matrix[user_id, item_id] = rating
        
        self.target = torch.tensor(rating_matrix)

        self.ids = torch.arange(self.num_users + self.num_items).long()

        del data
        del user_ids
        del item_ids
        del rating_matrix
    
    def get_num_user(self):
        return self.num_users
    
    def get_target(self):
        return self.target
    
    def get_input(self):
        return self.ids, self.edge_index



def get_data(config, mode):
    mode_posibilities = ['train', 'val', 'test', 'predict']

    assert mode in mode_posibilities, "Please chose a mode in " + str(mode_posibilities)

    data_path = os.path.join(config.data.path, mode + '.csv') if mode != 'predict' else config.predict.src_path
    num_items = config.data.num_items

    return Data(data_path, num_items)
