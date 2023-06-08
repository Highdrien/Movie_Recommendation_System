import pandas as pd

# Chargement du fichier CSV
df = pd.read_csv('Dataset.csv')

num_users = len(df['user_id'].unique())

SPLIT_1 = 0.6
SPLIT_2 = 0.2

split_1 = int(SPLIT_1 * num_users)
split_2 = int(SPLIT_2 * num_users) + split_1


# Filtrage des données pour chaque partie
partie1 = df[df['user_id'] < split_1]
partie2 = df[(df['user_id'] >= split_1) & (df['user_id'] <= split_2)]
partie3 = df[df['user_id'] >= split_2]

# Réindexer les user_id dans partie2 et partie3
partie2.loc[:, 'user_id'] = partie2['user_id'] - split_1
partie3.loc[:, 'user_id'] = partie3['user_id'] - split_2


# Sauvegarde dans des fichiers CSV
partie1.to_csv('train.csv', index=False)
partie2.to_csv('val.csv', index=False)
partie3.to_csv('test.csv', index=False)
