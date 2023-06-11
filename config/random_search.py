import os
import copy
import random

RANDOM_SEARCH_PATH = os.path.join('logs','random_search')
RANDOM_SEARCH_LOGS_PATH = os.path.join(RANDOM_SEARCH_PATH, 'random_search_logs.csv')
RS_DICO = {'learning_rate':     [0.01, 0.005, 0.001],
            'embedding_dim':    [0, 0, 0, 4, 8, 16, 32],
            'hidden_layer_1':   [4, 8, 16, 32, 64],
            'hidden_layer_2':   [8, 16, 32],
            'hidden_layer_3':   [1, 2, 4, 8],
            'dropout':          [0, 0.05, 0.1, 0.15],
            'middle_function':  ['relu', 'sigmoid', 'tanh', None],
            'end_function':     ['sigmoid', 'clamp', 'tanh', 'softmax', None]}


def get_random_value(array):
    return array[random.randint(0, len(array) -1)]


def generate_random_config(config):
    new_config = copy.deepcopy(config)
    new_config.train.logs_path = RANDOM_SEARCH_PATH

    for key, list_value in RS_DICO.items():
        value = get_random_value(list_value)
        print(key, ':', value)
        new_config.model[key] = value

    return new_config


def create_logs_file():
    if not os.path.exists(RANDOM_SEARCH_PATH):
        os.mkdir(RANDOM_SEARCH_PATH)

    with open(RANDOM_SEARCH_LOGS_PATH, 'x') as f:
        line = 'experiement_path,best_loss_value'
        for key, _ in RS_DICO.items():
            line += ',' + str(key)
        f.write(line + '\n')


def save_step(new_config, val_loss, path):
     with open(os.path.join(RANDOM_SEARCH_PATH, 'random_search_logs.txt'), 'a') as f:
        line = path + ',' + str(val_loss)
        for key, _ in RS_DICO.items():
            line += ',' + str(new_config.model[key])
        f.write(line + '\n')

