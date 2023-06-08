# Movie Recommendator System

# Requirements

To run the code you need python (We use python 3.9.13) and all the packages which your can find on the requirements.txt file. You can run the following code to install all packages in the correct versions:
```bash
pip install -r requirements.txt
```

# Run the code

To run the program, simply execute the `main.py` file. However, there are several modes.

## Mode: train

To do this, you need to choose a `.yaml` configuration file to set all the training parameters. By default, the code will use the `config/configs.yaml` file. The code will create a folder: 'experiment' in logs to store all the training information, such as a copy of the configuration used, the loss and metrics values at each epoch, the learning curves and the model weights.
To run a training session, enter the following command:
```bash
python main.py --mode train --config_path <path to your configuration system> 
```

## Mode: test

Test mode is used to evaluate an experiment on the test database. It will give you the value of the Loss in a file test_log.txt which will be located in the experiment folder.
```bash
python main.py --mode test --path <path to the experiment which you want to evaluate> 
```

## Mode: train and test

Train and Test mode is used to do a training and after do run a test on this training.
```bash
python main.py --mode train_and_test
```

## Mode: predict

The predict mode is used to predict all the images in the congif.predict.src_path folder. These images will be split into patches, then predicted, then reconstituted and finally saved in the folder indicated in config.predict.dst_path.

```bash
python main.py --mode predict --path <path to the experiment which you want to use for the prediction> 
```

