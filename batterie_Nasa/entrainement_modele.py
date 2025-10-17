# Fichier: entrainement_modele.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# --- 1. Charger les données préparées ---
try:
    data = pd.read_csv('donnees_batteries_preparees.csv')
    print("Données préparées chargées avec succès.")
except FileNotFoundError:
    print("ERREUR: Le fichier 'donnees_batteries_preparees.csv' est introuvable.")
    print("Veuillez d'abord exécuter le script de l'étape H.2 pour le générer.")
    exit()

# --- 2. Préparation des séquences ---
# Un LSTM a besoin de séquences de données (ex: regarder les 10 derniers cycles pour prédire le suivant)
sequence_length = 10  # Taille de la fenêtre de cycles à observer
features = ['capacity']   # Caractéristiques à utiliser pour la prédiction
target = 'RUL'            # Ce que nous voulons prédire

def create_sequences(df, seq_length, features, target):
    X, y = [], []
    for battery_id, group in df.groupby('battery_id'):
        feature_data = group[features].values
        target_data = group[target].values
        for i in range(len(group) - seq_length):
            X.append(feature_data[i:i + seq_length])
            y.append(target_data[i + seq_length - 1])
    return np.array(X), np.array(y)

X, y = create_sequences(data, sequence_length, features, target)

print(f"\nDonnées transformées en séquences de longueur {sequence_length}.")
print(f"Forme des données d'entrée (X): {X.shape}") # (nombre_sequences, sequence_length, nombre_features)
print(f"Forme des données de sortie (y): {y.shape}")   # (nombre_sequences)

# --- 3. Normalisation et division des données ---
# On normalise les données pour que le modèle apprenne plus efficacement
scaler = MinMaxScaler()
X = scaler.fit_transform(X.reshape(-1, len(features))).reshape(X.shape)

# Diviser en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nDonnées divisées en ensembles d'entraînement et de test.")

# --- 4. Construction du modèle LSTM ---
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, len(features))),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1) # Couche de sortie avec 1 neurone pour prédire la RUL
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# --- 5. Entraînement du modèle ---
print("\n--- Début de l'entraînement du modèle ---")
history = model.fit(
    X_train, y_train,
    epochs=50,  # Nombre de fois que le modèle va voir toutes les données
    batch_size=32,
    validation_split=0.1, # Utiliser 10% des données d'entraînement pour la validation
    verbose=1
)
print("--- Entraînement terminé ---")

# --- 6. Évaluation et Sauvegarde ---
loss = model.evaluate(X_test, y_test)
print(f"\nPerte sur l'ensemble de test (MSE): {loss:.4f}")

# Sauvegarder le modèle entraîné
nom_modele = 'rul_model.h5'
model.save(nom_modele)
print(f"\nModèle sauvegardé avec succès sous le nom: '{nom_modele}'")

# Sauvegarder le scaler pour l'utiliser lors de l'inférence
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler sauvegardé avec succès sous le nom: 'scaler.pkl'")

# --- 7. Visualisation des résultats (Optionnel) ---
predictions = model.predict(X_test).flatten()

plt.figure(figsize=(12, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Prédictions vs. Valeurs Réelles')
plt.xlabel('RUL Réel')
plt.ylabel('RUL Prédit')
plt.grid(True)
plt.show()