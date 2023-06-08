import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_learning_curves(path):
    result, names = get_result(path)

    epochs = result[:, 0]
    for i in range(1, len(names), 2):
        train_metrics = result[:, i]
        val_metrics = result[:, i + 1]
        plt.plot(epochs, train_metrics)
        plt.plot(epochs, val_metrics)
        plt.title(names[i])
        plt.xlabel('epoch')
        plt.ylabel(names[i])
        plt.legend(names[i:])
        plt.grid()
        plt.savefig(os.path.join(path, names[i] + '.png'))
        plt.close()


def get_result(path):
    with open(os.path.join(path, 'train_log.csv'), 'r') as f:
        names = f.readline()[:-1].split(',')
        result = []
        for line in f:
            result.append(line[:-1].split(','))

        result = np.array(result, dtype=float)
    f.close()
    return result, names


def find_5_best_film_to_see(csv_path, movie_title_path):

    df_ratings = pd.read_csv(csv_path)
    df_movies = pd.read_csv(movie_title_path)
    df_merged = pd.merge(df_ratings, df_movies, on='item_id', how='left')
    # df_sorted = df_merged.sort_values('rating', ascending=False).head(5)
    # print(df_sorted)
    for user_id in df_merged['user_id'].unique():
        top_movies_user_id = df_merged[df_merged['user_id'] == user_id]
        df_sorted = top_movies_user_id.sort_values('rating', ascending=False).head(5)
        print('\nBest film to see for the user:', user_id)
        # print(df_sorted)
        print(df_sorted[['user_id', 'rating', 'title', 'year']])





if __name__ == "__main__":
    csv_path = os.path.join('data', 'prediction', 'predict.csv')
    movie_title_path = os.path.join('data', 'Movie_Id_Titles.csv')
    find_5_best_film_to_see(csv_path, movie_title_path)