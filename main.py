import os
import yaml
import argparse
from easydict import EasyDict as edict

from src.train import train
from src.test import test


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


def main(options):
    if options['mode'] == 'train':
        config = load_config(options['config_path'])
        train(config)
    
    elif options['mode'] == 'test':
        config_path = os.path.join(options['path'], find_config(options['path']))
        config = load_config(config_path)
        test(options['path'], config)
    
    if options['mode'] == 'train_and_test':
        print('---train---')
        config = load_config(options['config_path'])
        logging_path = train(config)
        print('---test---')
        test(logging_path, config)

    else:
        print('ERROR: mode incorect. You chose: ' + options['mode'] + '. Please chose a mode between:')
        print('- train')
        print('- test')
        print('- train_and_test')
        exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--mode', default=None, type=str, help="choose a mode between 'train', 'test'")
    parser.add_argument('--config_path', default=os.path.join('config', 'config.yaml'), type=str, help="path to config (for training)")
    parser.add_argument('--path', type=str, help="experiment path (for test, prediction or resume previous training)")

    args = parser.parse_args()
    options = vars(args)

    main(options)