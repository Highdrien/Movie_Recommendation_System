# Movie Recommendator System

- [Movie Recommendator System](#movie-recommendator-system)
- [Requirements](#requirements)
- [Run the code](#run-the-code)
  - [Mode: train](#mode-train)
  - [Mode: test](#mode-test)
  - [Mode: train and test](#mode-train-and-test)
  - [Mode: random search](#mode-random-search)
  - [Mode: predict](#mode-predict)
- [Results](#results)


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

## Mode: random search
To do a random search on the parametre, you can select the parameter in the the file `configs/random_search.py` and run the following commend:
```bash
python main.py --mode random_search --steps <the number of training wwhich you want to do>
```
All the experiemnts will be save on the folder `logs/random_search`, and the file `random_search_logs.csv` will save all the parameter and the best validation loss for each training.

## Mode: predict

To make a prediction, you first have to enter a list of the films a person has seen and write them down. This can be a long and tedious process. To simplify this task, we made two python programs that generate a user interface. The first program: `find_film_vu.py` displays a window scrolling through the names of the films, with 2 yes/no buttons at the bottom. You then have to say whether or not you've seen these films. The results are then stored in a `films_vu.csv`. Next, we run the second program: `rate_films.py`, which scrolls through all the films we've seen and asks us to rate them between 1 and 5. This automatically saves everything in `rating.csv`, directly in a good format so that the algorithm can make a prediction. 

When you just have to run the following commend:
```bash
python main.py --mode predict --path <path to the experiment which you want to use for the prediction> 
```
This will create a new csv file: `predict_experiment_<number of your experiment>.csv` containing all the films and the predicted ratings. The 5 best films according to the algorithm will be displayed in the terminal.

You can see some statictic about the prediction with the programme `statistic.py` like the mean and the standart deviation of all the rating.

# Results

You can see the best experiment for the model 1, 2 and 3 in the folder respectively: `logs/random_search/experiment_35`, `logs/random_search/experiment_47`, `logs/random_search/experiment_11` (see the report `Movie_Recommendator_System.pdf` for more details)