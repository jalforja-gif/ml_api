# ml_api.py
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import xgboost as xgb
import numpy as np

app = Flask(__name__)

# ==============================
# LOAD MODELS
# ==============================

# ✅ Sentence-BERT for similarity
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ Real BERT classifier (you can fine-tune later)
# ✅ Real BERT classifier (public demo model)
bert_model_path = "models/bert_classifier"  # public model
tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_path)

# Label mapping
LABELS = [
    "Copyright",
    "Patent",
    "Trademark",
    "Utility Model",
    "Industrial Design"
]

# ✅ XGBoost readiness model (demo trained)
# ✅ XGBoost readiness model (demo)
xgb_model = xgb.XGBRegressor()
try:
    xgb_model.load_model("models/xgb_readiness.json")
except FileNotFoundError:
    print("Warning: xgb_readiness.json not found. Using fallback readiness predictor.")
    xgb_model = None

# ==============================
# BERT CLASSIFICATION (REAL)
# ==============================
def classify_ip_bert(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return LABELS[pred], probs[0].tolist()

# ==============================
# READINESS VIA XGBOOST
# ==============================
def predict_readiness(similarity_score):
    features = np.array([[similarity_score]])
    if xgb_model is not None:
        try:
            return float(xgb_model.predict(features)[0])
        except:
            pass
    # fallback if model missing or fails
    return round(1 - similarity_score, 4)

# ==============================
# API ENDPOINT
# ==============================
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    title = data.get('title', '')
    abstracts = data.get('existing_abstracts', [])

    # ==============================
    # 1. BERT CLASSIFICATION
    # ==============================
    suggested_classification, class_probs = classify_ip_bert(title)

    # ==============================
    # 2. SBERT SIMILARITY
    # ==============================
    title_embedding = similarity_model.encode(title, convert_to_tensor=True)

    max_similarity = 0.0
    for text in abstracts:
        emb = similarity_model.encode(text, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(title_embedding, emb).item()
        if sim > max_similarity:
            max_similarity = sim

    # ==============================
    # 3. XGBOOST READINESS
    # ==============================
    readiness = predict_readiness(max_similarity)

    return jsonify({
        "suggested_classification": suggested_classification,
        "classification_confidence": max(class_probs),
        "similarity_score": round(max_similarity, 4),
        "readiness_score": round(readiness, 4),
        "embedding": title_embedding.cpu().tolist()
    })

@app.route('/ping')
def ping():
    return "ML API is running!"

# ==============================
# RUN SERVER
# ==============================
# if __name__ == '__main__':

#    app.run(debug=True, port=5000)
