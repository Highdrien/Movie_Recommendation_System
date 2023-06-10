import os
import yaml
import argparse
from easydict import EasyDict as edict

from src.train import train
from src.test import test
from src.predict import predict
from config.random_search import generate_random_config, create_logs_file, save_step


def load_config(path='configs/config.yaml'):
    stream = open(path, 'r')
    return edict(yaml.safe_load(stream))


def find_config(experiment_path):
    yaml_in_path = list(filter(lambda x: x[-5:] == '.yaml', os.listdir(experiment_path)))
    
    if len(yaml_in_path) == 1:
        return yaml_in_path[0]

    if len(yaml_in_path) == 0:
        print("ERROR: config.yaml wasn't found in", experiment_path)
    
    if len(yaml_in_path) > 0:
        print("ERROR: a lot a .ymal was found in", experiment_path)
    
    exit()


def apply_random_search(config_path, steps):
    config = load_config(config_path)
    best_path, best_val_loss = "", 10e6
    create_logs_file()
    for step in range(steps):
        print('\n ------------------------')
        print('random search: iteration ' + str(step + 1) + '/' + str(steps))
        new_config = generate_random_config(config)
        path, val_loss = train(new_config)
        save_step(new_config, val_loss, path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = path
    print('random search done')
    print('best val loss was:', best_val_loss)
    print('it was save in:', best_path)


def check_options(options):
    """
    Check if all the options is good
    """
    mode_list = ['train', 'test', 'train_and_test', 'predict', 'random_search']
    default_config = os.path.join('configs', 'config.yaml')

    if options['mode'] not in mode_list:
        print('\nERROR: incorect mode. You chose: ' + str(options['mode']) + '. Please chose a mode between:')
        print(mode_list)
        print('with adding --mode <your_mode>')
        exit()
    
    if options['mode'] == 'train' and options['config_path'] is None:
        options['config_path'] = default_config
        print("You chose the config:", default_config)

    if options['mode'] in ['test', 'predict'] and options['path'] is None:
        print('ERROR: please chose an experiment path for your', options['mode'])
        exit()
    
    if options['mode'] == 'random_search' and options['steps'] is None:
        print('ERROR: please chose a steps to do your random search')
        exit()
    
    return options
   

def main(options):

    options = check_options(options)

    if options['mode'] == 'train':
        config = load_config(options['config_path'])
        train(config)
    
    elif options['mode'] == 'test':
        config_path = os.path.join(options['path'], find_config(options['path']))
        config = load_config(config_path)
        test(options['path'], config)
    
    elif options['mode'] == 'train_and_test':
        print('---train---')
        config = load_config(options['config_path'])
        logging_path = train(config)
        print('---test---')
        test(logging_path, config)
    
    elif options['mode'] == 'predict':
        config_path = os.path.join(options['path'], find_config(options['path']))
        config = load_config(config_path)
        predict(options['path'], config)
    
    elif options['mode'] == 'random_search':
        apply_random_search(options['config_path'], options['steps'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--mode', default=None, type=str, help="choose a mode between 'train', 'test', 'predict', or 'random_search'")
    parser.add_argument('--config_path', default=os.path.join('config', 'config.yaml'), type=str, help="path to config (for training)")
    parser.add_argument('--path', type=str, help="experiment path (for test, prediction or resume previous training)")
    parser.add_argument('--steps', type=int, help="number of iteration of a random search")

    args = parser.parse_args()
    options = vars(args)

    main(options)