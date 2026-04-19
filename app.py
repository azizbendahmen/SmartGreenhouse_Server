from flask import Flask, request, jsonify, render_template
from datetime import datetime
import requests
import time
import base64
import io
import cv2
import numpy as np
from google import genai
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════
# 🔐  GEMINI
# ══════════════════════════════════════════════════════════════
GEMINI_API_KEY = "AIzaSyBkzOgKi46g138Y_56pqKKcuuckVrEmKuM"
try:
    client_gemini = genai.Client(api_key=GEMINI_API_KEY)
except Exception:
    client_gemini = None

# ══════════════════════════════════════════════════════════════
# 🧠  IA LOCALE
# ══════════════════════════════════════════════════════════════
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Initialisation IA sur : {DEVICE}")

try:
    checkpoint = torch.load("plant_model.pth", map_location=DEVICE)
    CLASSES    = checkpoint["classes"]
    mobilenet  = models.mobilenet_v3_small()
    mobilenet.classifier[3] = nn.Linear(1024, len(CLASSES))
    mobilenet.load_state_dict(checkpoint["model"])
    mobilenet.to(DEVICE).eval()
    print(f"MobileNetV3 OK ({len(CLASSES)} classes)")
except Exception as e:
    print(f"MobileNet ERREUR : {e}")
    mobilenet = None
    CLASSES   = []

try:
    yolo = YOLO("yolov8n.pt")
    print("YOLOv8n OK")
except Exception as e:
    print(f"YOLO ERREUR : {e}")
    yolo = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

MALADIES_GRAVES = [
    "bacterial_spot", "early_blight", "late_blight",
    "leaf_mold", "mosaic_virus", "yellow_leaf_curl"
]
INTRUS_LABELS = ["person", "cat", "dog", "rat", "bird", "mouse", "bear"]

last_vision_result = {
    "disease": None, "confidence": 0, "sain": True,
    "action": "Aucune", "alerte": None, "detections": []
}

def classify(img):
    if not mobilenet:
        return "Modele non charge", 0
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
            action, alerte = "Traitement urgent", f"Maladie critique : {classe}"
        else:
            action, alerte = "Ventilation conseillee", f"Stress foliaire : {classe}"
    for d in detections:
        if any(x in d["label"].lower() for x in INTRUS_LABELS):
            action, alerte = "Declenchement alarme", f"Intrusion : {d['label'].upper()}"
            break
    return action, alerte

def annotate_image(img, classe, conf, detections, sain):
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w  = frame.shape[:2]

    # Bandeau diagnostic haut
    BAND_H    = max(44, int(h * 0.075))
    band_bgr  = (34, 139, 34) if sain else (60, 50, 220)
    cv2.rectangle(frame, (0, 0), (w, BAND_H), band_bgr, -1)

    label_text = f"{'SAIN' if sain else 'MALADIE'} - {classe.replace('_',' ').title()}  ({conf}%)"
    font       = cv2.FONT_HERSHEY_DUPLEX
    fs         = max(0.42, min(0.72, w / 950))
    ty         = int(BAND_H * 0.68)
    cv2.putText(frame, label_text, (13, ty+1), font, fs, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(frame, label_text, (13, ty),   font, fs, (255,255,255), 1, cv2.LINE_AA)

    # Bounding boxes YOLO
    for det in detections:
        bbox = det.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x, y, bw, bh = [int(v) for v in bbox]
        is_intrus = any(xi in det["label"].lower() for xi in INTRUS_LABELS)
        color     = (30, 30, 220) if is_intrus else (34, 197, 94)

        # Fond semi-transparent
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x+bw, y+bh), color, -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

        # Bordure
        cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 2)

        # Coins brackets
        cs = max(10, min(22, int(min(bw, bh) * 0.14)))
        for pts in [[(x,y+cs),(x,y),(x+cs,y)], [(x+bw-cs,y),(x+bw,y),(x+bw,y+cs)],
                    [(x,y+bh-cs),(x,y+bh),(x+cs,y+bh)], [(x+bw-cs,y+bh),(x+bw,y+bh),(x+bw,y+bh-cs)]]:
            for i in range(len(pts)-1):
                cv2.line(frame, pts[i], pts[i+1], color, 3, cv2.LINE_AA)

        # Label
        lbl  = f"{det['label']}  {det['conf']}%"
        fs2  = max(0.36, min(0.54, w / 1200))
        (tw, th), _ = cv2.getTextSize(lbl, font, fs2, 1)
        ly = max(y - 6, th + 8)
        cv2.rectangle(frame, (x, ly-th-6), (x+tw+10, ly+2), color, -1)
        cv2.putText(frame, lbl, (x+5, ly-3), font, fs2, (255,255,255), 1, cv2.LINE_AA)

        # Contour alarme rouge pour intrus
        if is_intrus:
            cv2.rectangle(frame, (x-3, y-3), (x+bw+3, y+bh+3), (0, 0, 255), 3)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode("utf-8")

# ══════════════════════════════════════════════════════════════
# 📊  DATA IoT & MÉTÉO
# ══════════════════════════════════════════════════════════════
last_api_call    = 0
prediction_pluie = False
weather_data = {
    "prevision_pluie": False, "temp_ext": None, "precip_mm": None,
    "wind_kmh": None, "humidity_ext": None, "pression_ext": None,
    "sunrise": None, "sunset": None,
    "hourly_precip_prob": [], "hourly_temp": [], "hourly_labels": []
}
latest_data = {
    "temperature": 0, "humidite": 0, "pression": 0, "gaz": 0,
    "sol": 0, "pluie": 0, "fan": 0, "pompe": 0,
    "timestamp": "--:--:--", "prevision_pluie": False
}
manual_commands = {"fan": None, "pump": None}
history_data    = {"labels": [], "temperature": [], "humidite": [], "sol": []}

def check_weather_prediction():
    global last_api_call, prediction_pluie, weather_data
    if time.time() - last_api_call > 3600 or last_api_call == 0:
        try:
            url = (
                "https://api.open-meteo.com/v1/forecast"
                "?latitude=36.8065&longitude=10.1815"
                "&daily=precipitation_sum,temperature_2m_max,windspeed_10m_max,"
                "precipitation_probability_max,relativehumidity_2m_max,"
                "surface_pressure_mean,sunrise,sunset"
                "&hourly=precipitation_probability,temperature_2m"
                "&current_weather=true&timezone=auto&forecast_days=2"
            )
            resp    = requests.get(url, timeout=8).json()
            daily   = resp.get("daily", {})
            hourly  = resp.get("hourly", {})
            current = resp.get("current_weather", {})
            precip_tomorrow  = (daily.get("precipitation_sum") or [0,0])[1]
            prediction_pluie = precip_tomorrow > 2.0
            now_h  = datetime.now().hour
            start  = min(now_h, max(0, len(hourly.get("time", [])) - 8))
            end    = start + 8
            weather_data = {
                "prevision_pluie":    prediction_pluie,
                "temp_ext":           round(current.get("temperature", 0), 1),
                "precip_mm":          round(precip_tomorrow, 1),
                "wind_kmh":           round(current.get("windspeed", 0), 1),
                "humidity_ext":       (daily.get("relativehumidity_2m_max") or [None])[0],
                "pression_ext":       round((daily.get("surface_pressure_mean") or [1013])[0]),
                "sunrise":            ((daily.get("sunrise") or [""])[1].split("T")[-1])
                                      if len(daily.get("sunrise") or []) > 1 else "--:--",
                "sunset":             ((daily.get("sunset") or [""])[1].split("T")[-1])
                                      if len(daily.get("sunset") or []) > 1 else "--:--",
                "hourly_precip_prob": (hourly.get("precipitation_probability") or [])[start:end],
                "hourly_temp":        [round(t,1) for t in (hourly.get("temperature_2m") or [])[start:end]],
                "hourly_labels":      [t.split("T")[-1][:5] for t in (hourly.get("time") or [])[start:end]],
            }
            last_api_call = time.time()
            print(f"Meteo OK — pluie={prediction_pluie}, temp={weather_data['temp_ext']}C")
        except Exception as e:
            print(f"Meteo API erreur : {e}")
            prediction_pluie = False
    return prediction_pluie

# ══════════════════════════════════════════════════════════════
# 🌐  ROUTES
# ══════════════════════════════════════════════════════════════
@app.route('/')
def index(): return render_template('index.html')

@app.route('/data', methods=['POST'])
def receive_data():
    global latest_data, history_data
    try:
        data  = request.get_json()
        now   = datetime.now().strftime("%H:%M:%S")
        rain  = check_weather_prediction()
        fan_v  = manual_commands['fan']  if manual_commands['fan']  is not None else data.get("fan", 0)
        pump_v = manual_commands['pump'] if manual_commands['pump'] is not None else data.get("pompe", 0)
        latest_data = {
            "temperature": data.get("temperature",0), "humidite": data.get("humidite",0),
            "pression":    data.get("pression",0),    "gaz":      data.get("gaz",0),
            "sol":         data.get("sol",0),          "pluie":    data.get("pluie",0),
            "fan": fan_v, "pompe": pump_v, "timestamp": now, "prevision_pluie": rain
        }
        history_data["labels"].append(now)
        history_data["temperature"].append(latest_data["temperature"])
        history_data["humidite"].append(latest_data["humidite"])
        history_data["sol"].append(latest_data["sol"])
        if len(history_data["labels"]) > 20:
            for k in history_data: history_data[k].pop(0)
        return jsonify({"status": "success", "block_pump": rain}), 200
    except Exception: return jsonify({"status": "error"}), 400

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({"latest": latest_data, "history": history_data, "weather": weather_data})

@app.route('/api/weather', methods=['GET'])
def get_weather(): return jsonify(weather_data)

@app.route('/api/agronome', methods=['GET'])
def ask_agronomist():
    if not client_gemini:
        return jsonify({"conseil": "Erreur : cle API Gemini introuvable."})

    # Contexte Vision IA
    vision_ctx = "Aucune analyse visuelle effectuee pour cette session."
    if last_vision_result["disease"]:
        if last_vision_result["sain"]:
            vision_ctx = "Vision IA : plante saine, aucune pathologie detectee."
        else:
            vision_ctx = (
                f"Vision IA : pathologie foliaire detectee — {last_vision_result['disease']} "
                f"(confiance {last_vision_result['confidence']}%). "
                f"Action systeme : {last_vision_result['action']}."
            )
        intrus = [d["label"] for d in last_vision_result["detections"]
                  if any(x in d["label"].lower() for x in INTRUS_LABELS)]
        if intrus:
            vision_ctx += f" Intrusion detectee : {', '.join(intrus)}."

    prompt = f"""
Tu es un ingenieur agronome expert en serres intelligentes IoT.

DONNEES CAPTEURS IoT :
- Temperature serre : {latest_data['temperature']}C
- Humidite air      : {latest_data['humidite']}%
- Humidite sol      : {latest_data['sol']}%
- Pression          : {latest_data['pression']} hPa
- Qualite air (COV) : {latest_data['gaz']} kOhm
- Pluie exterieure  : {'Detectee' if latest_data['pluie'] else 'Aucune'}
- Ventilateur       : {'ON' if latest_data['fan']==1 else 'OFF'}
- Pompe irrigation  : {'ON' if latest_data['pompe']==1 else 'OFF'}
- Prevision pluie demain : {'OUI - irrigation suspendue' if latest_data['prevision_pluie'] else 'NON'}

VISION IA (MobileNetV3 + YOLOv8) :
{vision_ctx}

Redige un rapport d audit structure en 4 points :
1. Diagnostic global : etat general de la serre (1-2 phrases).
2. Analyse pathologique : si maladie detectee, explique precisement la pathologie, ses causes, ses symptomes visuels, son niveau de risque pour la culture.
3. Recommandations de traitement : protocole precis avec produit(s) recommande(s), dosage, frequence d application, conditions environnementales a ajuster (temperature, humidite, ventilation).
4. Alertes prioritaires : actions immediates a prendre dans les 24 prochaines heures.

Style : professionnel, direct, chiffres concrets. Format markdown avec titres en gras.
"""
    try:
        resp = client_gemini.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return jsonify({"conseil": resp.text})
    except Exception:
        return jsonify({"conseil": "Erreur de connexion au serveur Gemini."})

@app.route('/api/analyze_plant', methods=['POST'])
def analyze_plant():
    global last_vision_result
    if 'file' not in request.files:
        return jsonify({"error": "Aucune image recue"}), 400
    try:
        img_bytes = request.files['file'].read()
        img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_w, img_h = img.size

        # 1. MobileNetV3
        classe, conf = classify(img)
        sain = "healthy" in classe.lower()

        # 2. YOLOv8
        detections = []
        if yolo:
            res = yolo(img, verbose=False)[0]
            for b in res.boxes:
                if float(b.conf) < 0.45: continue
                label = res.names[int(b.cls)].upper()
                c     = round(float(b.conf)*100, 1)
                x1,y1,x2,y2 = [float(v) for v in b.xyxy[0]]
                detections.append({
                    "label": label, "conf": c,
                    "bbox": [x1, y1, x2-x1, y2-y1], "bboxNorm": False
                })

        # 3. Decision
        action, alerte = decide(classe, conf, detections)

        # 4. Annotation OpenCV
        annotated_b64 = annotate_image(img, classe, conf, detections, sain)

        # 5. Sauvegarde pour Audit IA
        last_vision_result = {
            "disease": classe.replace("_"," ").capitalize(),
            "confidence": conf, "sain": sain,
            "action": action, "alerte": alerte, "detections": detections
        }

        return jsonify({
            "status":        "success",
            "disease":       classe.replace("_"," ").capitalize(),
            "confidence":    conf,
            "sain":          sain,
            "action":        action,
            "alerte":        alerte,
            "detections":    detections,
            "imageWidth":    img_w,
            "imageHeight":   img_h,
            "annotated_img": annotated_b64,
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/command', methods=['POST'])
def set_command():
    global manual_commands
    try:
        data   = request.get_json()
        device = data.get('device')
        mode   = data.get('mode')
        state  = data.get('state')
        if device not in ['fan','pump']:
            return jsonify({'status':'error','msg':'device invalide'}), 400
        if mode == 'auto':
            manual_commands[device] = None
            return jsonify({'status':'ok','mode':'auto','device':device})
        elif mode == 'manual':
            manual_commands[device] = int(state) if state is not None else 0
            return jsonify({'status':'ok','mode':'manual','device':device,'state':manual_commands[device]})
        return jsonify({'status':'error','msg':'mode invalide'}), 400
    except Exception as e:
        return jsonify({'status':'error','msg':str(e)}), 400

@app.route('/api/commands', methods=['GET'])
def get_commands():
    return jsonify({'fan': manual_commands['fan'], 'pump': manual_commands['pump']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=False)
