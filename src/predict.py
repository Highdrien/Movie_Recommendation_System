import os
import torch

from src.model import get_model
from src.dataloader import get_data
from src.checkpoints import get_checkpoint_path
from src.utils import find_5_best_film_to_see


def predict(logging_path, config):    

    # Get data
    predict_data = get_data(config, 'predict')
    predict_num_users = predict_data.get_num_user()
    predict_ids, predict_edge_index = predict_data.get_input()
    predict_target = predict_data.get_target()

    # Get model and load model's weight
    model = get_model(config)
    checkpoint_path = get_checkpoint_path(config, logging_path)
    model.load_state_dict(torch.load(checkpoint_path))

    model.eval()

    with torch.no_grad():
        model.eval()
        prediction = model(predict_ids, predict_edge_index, predict_num_users)
    
    dst_path = config.predict.dst_path[:-4] + '_' + os.path.basename(logging_path.rstrip(os.sep)) + '.csv'
    print(dst_path)
    save_prediction(prediction, predict_target, dst_path)
    find_5_best_film_to_see(dst_path, config.data.movie_title_path)

    
def save_prediction(prediction, predict_target, dst_path):
    """
    save the prediction on a csv file
    """
    with open(dst_path, 'w') as f:
        headers = "user_id,item_id,rating\n"
        f.write(headers)
        n, m = prediction.shape
        for i in range(n):
            for j in range(m):
                if predict_target[i, j] == 0:
                    f.write(str(i) + ',' + str(j) + ',' + str(prediction[i, j].item()) + '\n')
    f.close()
    print('prediction: done')
