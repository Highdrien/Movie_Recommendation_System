import csv
from tkinter import Tk, Label, Button, Frame

# Chemin du fichier CSV contenant la liste des films vus
fichier_films_vus = 'films_vu.csv'
# Chemin du fichier CSV pour enregistrer les notes des films
fichier_notes = 'ratings.csv'
user_id = 0

# Définition des fonctions pour les boutons de notation
def attribuer_note(note):
    global index
    film = films[index]
    notes[film[0]] = note
    print(f"Film ID : {film[0]}, Note : {note}")
    sauvegarder_notes()
    afficher_film_suivant()

def afficher_film_suivant():
    global index
    index += 1

    # Si tous les films ont été parcourus, fermer la fenêtre
    if index >= len(films):
        fenetre.destroy()
        sauvegarder_notes()
        return

    # Afficher le film suivant
    film_label.config(text=films[index][1])

# Fonction pour sauvegarder les notes des films dans un fichier CSV
def sauvegarder_notes():
    with open(fichier_notes, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['user_id', 'item_id', 'rating'])
        for film_id, note in notes.items():
            writer.writerow([user_id, film_id, note])

# Lecture du fichier CSV contenant la liste des films vus
with open(fichier_films_vus, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    films = [(row['item_id'], row['title']) for row in reader]

# Initialisation des variables
notes = {}
index = 0

# Création de la fenêtre
fenetre = Tk()
fenetre.title("Notation des films")

# Frame principale pour les widgets
frame = Frame(fenetre, padx=10, pady=10)
frame.pack()

# Création des widgets
film_label = Label(frame, text=films[index][1], wraplength=300)
film_label.pack()

boutons_frame = Frame(frame)
boutons_frame.pack()

# Création des boutons de notation
for i in range(1, 6):
    bouton_note = Button(boutons_frame, text=str(i), width=10, command=lambda note=i: attribuer_note(note))
    bouton_note.pack(side='left', padx=5, pady=5)

# Affichage de la fenêtre
fenetre.mainloop()
