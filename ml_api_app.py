# ml_api_app.py
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import xgboost as xgb
import numpy as np
import os

app = Flask(__name__)

# ==============================
# LOAD MODELS ONCE
# ==============================
print("Loading ML models...")

# SBERT (use cache folder if needed)
similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# BERT classifier
bert_model_path = os.path.join("models", "bert_classifier")
tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_path)

LABELS = ["Copyright", "Patent", "Trademark", "Utility Model", "Industrial Design"]

# XGBoost (optional)
xgb_model_path = os.path.join("models", "xgb_readiness.json")
xgb_model = None
if os.path.exists(xgb_model_path):
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(xgb_model_path)
else:
    print("Warning: xgb_readiness.json not found. Using fallback readiness predictor.")

print("Models loaded successfully!")

# ==============================
# UTIL FUNCTIONS
# ==============================
def classify_ip_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return LABELS[pred], probs[0].tolist()

def predict_readiness(similarity_score):
    features = np.array([[similarity_score]])
    if xgb_model:
        try:
            return float(xgb_model.predict(features)[0])
        except:
            pass
    return round(1 - similarity_score, 4)

# ==============================
# API ENDPOINTS
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    title = data.get("title", "")
    abstracts = data.get("existing_abstracts", [])

    # BERT classification
    suggested_classification, class_probs = classify_ip_bert(title)

    # SBERT similarity
    title_emb = similarity_model.encode(title, convert_to_tensor=True)
    max_similarity = 0.0
    for text in abstracts:
        emb = similarity_model.encode(text, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(title_emb, emb).item()
        if sim > max_similarity:
            max_similarity = sim

    # XGBoost readiness
    readiness = predict_readiness(max_similarity)

    return jsonify({
        "suggested_classification": suggested_classification,
        "classification_confidence": max(class_probs),
        "similarity_score": round(max_similarity, 4),
        "readiness_score": round(readiness, 4),
        "embedding": title_emb.cpu().tolist()
    })

@app.route("/ping")
def ping():
    return "ML API is running!"

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
