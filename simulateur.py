import requests
import time
import random

URL = "http://127.0.0.1:3000/data"

print("==========================================")
print("  SIMULATEUR IOT (EDGE DEVICE) ACTIF ")
print("==========================================")

temp, hum, pression, gaz, sol, pluie = 26.0, 55.0, 1013, 50.0, 85, 0
fan, pompe, ia_blocage = 0, 0, False

while True:
    temp += random.uniform(-0.2, 0.8) 
    hum += random.uniform(-1.0, 1.0)
    sol -= random.randint(2, 6) 

    if temp > 35.0: temp = 25.0 
    if hum > 100.0: hum = 100.0
    if hum < 20.0: hum = 20.0
    if sol < 0: sol = 0

    # Automates Edge
    if temp >= 30.0: fan = 1
    elif temp <= 27.0: fan = 0

    if sol < 30 and pluie == 0 and not ia_blocage:
        pompe = 1
        sol = 100
    else:
        pompe = 0

    payload = {
        "temperature": round(temp, 1), 
        "humidite": round(hum, 1), 
        "pression": pression, 
        "gaz": round(gaz, 1), 
        "sol": sol, 
        "pluie": pluie, 
        "fan": fan, 
        "pompe": pompe
    }

    try:
        res = requests.post(URL, json=payload)
        if res.status_code == 200:
            ia_blocage = res.json().get("block_pump", False)
            print(f"ENVOI -> Temp: {payload['temperature']}C | Sol: {payload['sol']}% | Fan: {fan} | Pump: {pompe}")
        else:
            print(f"Erreur HTTP : {res.status_code}")
    except requests.exceptions.ConnectionError:
        print("ATTENTION : Serveur Backend (app.py) injoignable. Lancez app.py d'abord.")

    time.sleep(2.5)