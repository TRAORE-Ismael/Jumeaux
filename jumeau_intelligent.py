# Fichier: jumeau_intelligent.py

import paho.mqtt.client as mqtt
import time
import json
import numpy as np
import tensorflow as tf
import pickle

# --- Paramètres de la batterie ---
CAPACITE_NOMINALE_Ah = 2.2

# --- Configuration MQTT ---
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_TELEMETRIE = "batterie/telemetrie"
TOPIC_JUMEAU_SOC = "batterie/jumeau/soc"
TOPIC_JUMEAU_SOH = "batterie/jumeau/soh"
TOPIC_JUMEAU_RUL = "batterie/jumeau/rul"  # NOUVEAU topic pour la prédiction RUL

# --- Chargement du modèle de prédiction et du scaler ---
try:
    model = tf.keras.models.load_model('rul_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Modèle de prédiction 'rul_model.h5' et 'scaler.pkl' chargés avec succès.")
except Exception as e:
    print(f"ERREUR: Impossible de charger le modèle ou le scaler. {e}")
    model = None
    scaler = None

# --- Variables d'état du Jumeau Numérique ---
soc = 100.0
soh = 100.0
rul_predit = -1 # Valeur initiale, -1 signifie "pas encore calculé"
derniere_mesure_temps = time.time()

# --- Variables pour le calcul du SoH et RUL ---
is_in_cycle_decharge = False
charge_accumulee_Ah = 0.0
historique_capacites = [] # NOUVEAU: pour stocker les capacités des derniers cycles
SEQUENCE_LENGTH = 10 # Doit être la même que lors de l'entraînement

# --- Fonctions de rappel MQTT ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connecté au broker MQTT !")
        client.subscribe(TOPIC_TELEMETRIE)
        print(f"Abonné au topic: {TOPIC_TELEMETRIE}")
    else:
        print(f"Échec de la connexion, code d'erreur: {rc}")

def on_message(client, userdata, msg):
    global soc, soh, rul_predit, derniere_mesure_temps, is_in_cycle_decharge, charge_accumulee_Ah, historique_capacites
    
    # ... (le début de la fonction on_message ne change pas) ...
    try:
        data = json.loads(msg.payload.decode())
        courant = data['courant']
        temps_actuel = time.time()
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Erreur de décodage du message: {e}")
        return
    dt_secondes = temps_actuel - derniere_mesure_temps
    if dt_secondes == 0: return
    derniere_mesure_temps = temps_actuel
    
    # Logique de réinitialisation du SoC
    if soc < 5 and courant > 0.5:
        soc = 100.0
        print("\n[INFO JUMEAU] Début de charge détecté. Réinitialisation du SoC à 100%.")

    # Calcul du SoC
    delta_charge_Ah = (courant * dt_secondes) / 3600.0
    delta_soc = (delta_charge_Ah / CAPACITE_NOMINALE_Ah) * 100.0
    soc += delta_soc
    soc = max(0.0, min(100.0, soc))
    
    # Logique de calcul du SoH et RUL (à la fin d'un cycle)
    if soc > 98 and courant < -1.0 and not is_in_cycle_decharge:
        is_in_cycle_decharge = True
        charge_accumulee_Ah = 0.0
        print("\n[INFO JUMEAU] Début de la détection d'un cycle de décharge.")

    if is_in_cycle_decharge and courant < 0:
        charge_accumulee_Ah += abs(delta_charge_Ah)

    if soc < 2 and is_in_cycle_decharge:
        is_in_cycle_decharge = False
        capacite_mesuree = charge_accumulee_Ah
        
        if capacite_mesuree > 0:
            soh = (capacite_mesuree / CAPACITE_NOMINALE_Ah) * 100.0
            soh = max(0.0, min(100.0, soh))
            
            print(f"\n[INFO JUMEAU] Fin du cycle détecté !")
            print(f"[INFO JUMEAU] Capacité mesurée: {capacite_mesuree:.3f} Ah")
            print(f"[INFO JUMEAU] NOUVEAU SOH CALCULÉ: {soh:.2f} %")
            client.publish(TOPIC_JUMEAU_SOH, f'{{"soh": {soh:.2f}}}')
            
            # --- NOUVEAU: Logique de prédiction de RUL ---
            # On ajoute la dernière capacité mesurée à notre historique
            historique_capacites.append(capacite_mesuree)
            
            # On ne fait une prédiction que si on a assez de données dans notre historique
            if len(historique_capacites) >= SEQUENCE_LENGTH and model is not None:
                # 1. On prend les N dernières capacités
                sequence_brute = np.array(historique_capacites[-SEQUENCE_LENGTH:]).reshape(-1, 1)
                
                # 2. On applique la MÊME normalisation que lors de l'entraînement
                sequence_normalisee = scaler.transform(sequence_brute)
                
                # 3. Le modèle s'attend à une forme (1, sequence_length, n_features)
                sequence_pour_prediction = np.expand_dims(sequence_normalisee, axis=0)
                
                # 4. On fait la prédiction !
                prediction = model.predict(sequence_pour_prediction)[0][0]
                rul_predit = max(0, int(prediction)) # On prend l'entier et on s'assure qu'il n'est pas négatif
                
                print(f"[INFO JUMEAU] NOUVELLE PRÉDICTION RUL: {rul_predit} cycles restants")
                client.publish(TOPIC_JUMEAU_RUL, f'{{"rul": {rul_predit}}}')

    # --- Publications ---
    print(f"SoC: {soc:6.2f} % | SoH: {soh:6.2f} % | RUL: {rul_predit if rul_predit != -1 else 'N/A'} cycles", end='\r')
    client.publish(TOPIC_JUMEAU_SOC, f'{{"soc": {soc:.2f}}}')

# --- Programme principal ---
# ... (le reste du script est identique) ...
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()
except KeyboardInterrupt:
    print("\nArrêt du Jumeau Numérique.")
    client.disconnect()
except Exception as e:
    print(f"Une erreur est survenue: {e}")