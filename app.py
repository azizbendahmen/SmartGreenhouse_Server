from flask import Flask, request, jsonify, render_template
from datetime import datetime
import requests
import time
from google import genai
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# ==========================================
# 🔐 CONFIGURATION GEMINI (LLM)
# ==========================================
GEMINI_API_KEY = "AIzaSyBkzOgKi46g138Y_56pqKKcuuckVrEmKuM"
try:
    client_gemini = genai.Client(api_key=GEMINI_API_KEY)
except:
    client_gemini = None

# ==========================================
# 🧠 CONFIGURATION IA LOCALE (PyTorch + YOLO)
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Initialisation de l'IA Locale sur : {DEVICE}")

try:
    checkpoint = torch.load("plant_model.pth", map_location=DEVICE)
    CLASSES = checkpoint["classes"]
    mobilenet = models.mobilenet_v3_small()
    mobilenet.classifier[3] = nn.Linear(1024, len(CLASSES))
    mobilenet.load_state_dict(checkpoint["model"])
    mobilenet.to(DEVICE).eval()
    print(f"✅ MobileNet chargé ({len(CLASSES)} classes)")
except Exception as e:
    print(f"❌ Erreur MobileNet (Avez-vous bien mis plant_model.pth ?) : {e}")
    mobilenet = None

try:
    yolo = YOLO("yolov8n.pt")
    print("✅ YOLOv8 chargé")
except Exception as e:
    print(f"❌ Erreur YOLO: {e}")
    yolo = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

MALADIES_GRAVES = ["bacterial_spot", "early_blight", "late_blight", "leaf_mold", "mosaic_virus", "yellow_leaf_curl"]

def classify(img):
    if not mobilenet: return "Modèle non chargé", 0
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(mobilenet(tensor), dim=1)[0]
    idx = probs.argmax().item()
    return CLASSES[idx], round(probs[idx].item() * 100, 1)

def decide(classe, conf, detections):
    action, alerte = "Aucune", None
    nom = classe.lower()

    if conf > 75 and "healthy" not in nom:
        if any(m in nom for m in MALADIES_GRAVES):
            action, alerte = "Traitement urgent", f"Maladie critique ({classe})"
        else:
            action, alerte = "Ventilation conseillée", f"Stress foliaire ({classe})"

    for d in detections:
        label = d["label"].lower()
        if any(x in label for x in ["person", "cat", "dog", "rat", "bird"]):
            action, alerte = "Déclenchement alarme", f"Intrusion : {d['label'].upper()}"

    return action, alerte

# ==========================================
# 📊 DATA IOT & MÉTÉO
# ==========================================
last_api_call = 0
prediction_pluie = False
latest_data = {"temperature": 0, "humidite": 0, "pression": 0, "gaz": 0, "sol": 0, "pluie": 0, "fan": 0, "pompe": 0, "timestamp": "--:--:--", "prevision_pluie": False}
history_data = {"labels": [], "temperature": [], "sol": []}

def check_weather_prediction():
    global last_api_call, prediction_pluie
    if time.time() - last_api_call > 3600 or last_api_call == 0:
        try:
            url = "https://api.open-meteo.com/v1/forecast?latitude=36.8065&longitude=10.1815&daily=precipitation_sum&timezone=auto"
            prediction_pluie = requests.get(url).json()['daily']['precipitation_sum'][1] > 2.0
            last_api_call = time.time()
        except:
            prediction_pluie = False
    return prediction_pluie

# ==========================================
# 🌐 ROUTES API
# ==========================================
@app.route('/')
def index(): return render_template('index.html')

@app.route('/data', methods=['POST'])
def receive_data():
    global latest_data, history_data
    try:
        data = request.get_json()
        now = datetime.now().strftime("%H:%M:%S")
        will_rain_tomorrow = check_weather_prediction()

        latest_data = {
            "temperature": data.get("temperature", 0), "humidite": data.get("humidite", 0),
            "pression": data.get("pression", 0), "gaz": data.get("gaz", 0),
            "sol": data.get("sol", 0), "pluie": data.get("pluie", 0),
            "fan": data.get("fan", 0), "pompe": data.get("pompe", 0),
            "timestamp": now, "prevision_pluie": will_rain_tomorrow
        }

        history_data["labels"].append(now)
        history_data["temperature"].append(latest_data["temperature"])
        history_data["sol"].append(latest_data["sol"])
        if len(history_data["labels"]) > 20:
            for key in history_data: history_data[key].pop(0)

        return jsonify({"status": "success", "block_pump": will_rain_tomorrow}), 200
    except:
        return jsonify({"status": "error"}), 400

@app.route('/api/data', methods=['GET'])
def get_data(): return jsonify({"latest": latest_data, "history": history_data})

@app.route('/api/agronome', methods=['GET'])
def ask_agronomist():
    if not client_gemini: return jsonify({"conseil": "Erreur : Clé API Gemini introuvable."})
    prompt = f"""
    CONTEXTE : Système de diagnostic d'une Serre Intelligente IoT. 
    Capteurs: BME680, Capacitif.
    DONNÉES ACTUELLES : Température : {latest_data['temperature']}°C, Humidité air : {latest_data['humidite']}%, Humidité sol : {latest_data['sol']}%. Pluie prévue demain : {'OUI' if latest_data['prevision_pluie'] else 'NON'}. Ventilateur : {'ON' if latest_data['fan']==1 else 'OFF'}. Pompe : {'ON' if latest_data['pompe']==1 else 'OFF'}.
    TÂCHE : Rédige un audit technique et agronomique (2 phrases maximum). Sois direct et professionnel (style ingénieur).
    """
    try:
        response = client_gemini.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return jsonify({"conseil": response.text})
    except Exception as e:
        return jsonify({"conseil": "Erreur de connexion au serveur Cloud."})

@app.route('/api/analyze_plant', methods=['POST'])
def analyze_plant():
    if 'file' not in request.files: return jsonify({"error": "Aucune image reçue"}), 400
    try:
        img_bytes = request.files['file'].read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        classe, conf = classify(img)
        detections = []
        if yolo:
            res = yolo(img, verbose=False)[0]
            detections = [{"label": res.names[int(b.cls)].upper(), "conf": round(float(b.conf)*100, 1)} for b in res.boxes if float(b.conf) > 0.5]

        action, alerte = decide(classe, conf, detections)

        return jsonify({
            "status": "success",
            "disease": classe.replace('_', ' ').capitalize(),
            "confidence": conf,
            "sain": "healthy" in classe.lower(),
            "action": action,
            "alerte": alerte,
            "detections": detections
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=False)