# Fichier: simulateur.py

import paho.mqtt.client as mqtt
import time
import json

# --- Configuration MQTT ---
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "batterie/telemetrie"

client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)

# --- Paramètres de simulation ---
facteur_degradation_par_cycle = 0.995 # La batterie perd 0.5% de sa capacité à chaque cycle
capacite_actuelle_Ah = 2.2 # Capacité de départ (identique à la capacité nominale)
courant_decharge_A = -1.5  # Courant de décharge constant
courant_charge_A = 1.0     # Courant de charge constant

cycle_count = 0

print("Démarrage du simulateur de batterie avec vieillissement...")
print("Appuyez sur Ctrl+C pour arrêter.")

try:
    # Boucle infinie pour simuler plusieurs cycles
    while True:
        cycle_count += 1
        print(f"\n--- DÉBUT CYCLE N°{cycle_count} | Capacité actuelle: {capacite_actuelle_Ah:.3f} Ah ---")
        
        # --- PHASE DE DÉCHARGE ---
        print("Phase de décharge...")
        tension = 4.2
        temperature = 25.0
        # La durée de la décharge dépend maintenant de la capacité réelle
        duree_decharge_secondes = (capacite_actuelle_Ah / abs(courant_decharge_A)) * 3600 
        pas_temps = 2 # Envoi des données toutes les 2 secondes
        nombre_etapes = int(duree_decharge_secondes / pas_temps)
        baisse_tension_par_etape = (4.2 - 3.0) / nombre_etapes

        for _ in range(nombre_etapes):
            tension -= baisse_tension_par_etape
            temperature += 0.01

            data = {"tension": round(tension, 3), "courant": courant_decharge_A, "temperature": round(temperature, 2)}
            client.publish(MQTT_TOPIC, json.dumps(data))
            # Affiche sur une seule ligne qui se met à jour
            print(f"Décharge... Tension: {data['tension']:.2f} V", end='\r')
            time.sleep(pas_temps)

        print("\n--- FIN DE DÉCHARGE ---")
        
        # --- PHASE DE CHARGE (simulée rapidement) ---
        print("--- DÉBUT DE CHARGE ---")
        # On simule une charge en envoyant quelques messages avec un courant positif
        for i in range(10):
            tension = 3.0 + (i * 0.12)
            data = {"tension": round(tension, 2), "courant": courant_charge_A, "temperature": 28.0}
            client.publish(MQTT_TOPIC, json.dumps(data))
            time.sleep(1)
        
        # La batterie est "pleine" pour le prochain cycle
        print("--- BATTERIE CHARGÉE ---")
        
        # On applique le vieillissement pour le prochain cycle
        capacite_actuelle_Ah *= facteur_degradation_par_cycle
        time.sleep(5) # Petite pause entre les cycles

except KeyboardInterrupt:
    print("\nSimulation arrêtée.")
finally:
    client.disconnect()