import os
import csv
from tkinter import Tk, Label, Button, Frame

# Chemin du fichier CSV contenant la liste des films
fichier_films = os.path.join('..', 'Movie_Id_Titles.csv')
fichier_sortie = 'films_vu.csv'

# Définition des fonctions pour les boutons Oui/Non
def film_vu():
    global index
    film = films[index]
    films_vus.append(film)
    print(f"Vous avez vu : {film}")
    sauvegarder_films_vus()
    afficher_film_suivant()

def film_non_vu():
    afficher_film_suivant()

def afficher_film_suivant():
    global index
    index += 1

    # Si tous les films ont été parcourus, fermer la fenêtre
    if index >= len(films):
        fenetre.destroy()
        sauvegarder_films_vus()
        return

    # Afficher le film suivant
    film_label.config(text=films[index])

# Fonction pour sauvegarder la liste des films vus dans un fichier CSV
def sauvegarder_films_vus():
    with open(fichier_sortie, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['item_id', 'title'])
        for film in films_vus:
            writer.writerow(film)

# Lecture du fichier CSV contenant la liste des films
with open(fichier_films, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    films = [(row['item_id'], row['title']) for row in reader]

# Initialisation des variables
films_vus = []
index = 0

# Création de la fenêtre
fenetre = Tk()
fenetre.title("Films")

# Frame principale pour les widgets
frame = Frame(fenetre, padx=10, pady=10)
frame.pack()

# Création des widgets
film_label = Label(frame, text=films[index], wraplength=300)
film_label.pack()

boutons_frame = Frame(frame)
boutons_frame.pack()

oui_bouton = Button(boutons_frame, text="Oui", width=10, command=film_vu)
oui_bouton.pack(side='left', padx=5, pady=5)

non_bouton = Button(boutons_frame, text="Non", width=10, command=film_non_vu)
non_bouton.pack(side='left', padx=5, pady=5)

# Affichage de la fenêtre
fenetre.mainloop()