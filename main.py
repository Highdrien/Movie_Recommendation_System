import os
import yaml
import argparse
from easydict import EasyDict as edict

from src.train import train
from src.test import test
from src.predict import predict


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


def check_options(options):
    """
    Check if all the options is good
    """
    mode_list = ['train', 'test', 'train_and_test', 'predict']
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--mode', default=None, type=str, help="choose a mode between 'train', 'test'")
    parser.add_argument('--config_path', default=os.path.join('config', 'config.yaml'), type=str, help="path to config (for training)")
    parser.add_argument('--path', type=str, help="experiment path (for test, prediction or resume previous training)")

    args = parser.parse_args()
    options = vars(args)

    main(options)