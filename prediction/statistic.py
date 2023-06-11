import pandas as pd

# Lecture des données à partir du fichier CSV
data = pd.read_csv("predict_experiment_47.csv")

# Calcul de la moyenne des évaluations (ratings)
moyenne = data["rating"].mean()

# Calcul de la variance des évaluations (ratings)
std = data["rating"].std()

# Affichage des résultats
print("Moyenne des évaluations :", moyenne)
print("Variance des évaluations :", std)
