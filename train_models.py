# train_models_db.py
import os
import torch
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pymysql  # for MySQL
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# 1. LOAD DATA FROM DATABASE
# -----------------------------
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='',
    database='ip_monitoring'  # <- your database name
)

# Fetch all IP data
ip_data = pd.read_sql("SELECT title, classification FROM intellectual_properties", conn)

# Example for XGBoost: generate placeholder similarity/readiness scores
xgb_data = pd.DataFrame({
    'similarity_score': np.random.rand(len(ip_data)),  # placeholder, can compute real similarity later
    'readiness_score': np.random.rand(len(ip_data))    # placeholder
})

conn.close()

# -----------------------------
# 2. TRAIN XGBOOST
# -----------------------------
xgb_model_file = os.path.join(BASE_DIR, "models/xgb_readiness.json")
os.makedirs(os.path.dirname(xgb_model_file), exist_ok=True)

xgb_model = xgb.XGBRegressor()
xgb_model.fit(xgb_data[['similarity_score']], xgb_data['readiness_score'])
xgb_model.save_model(xgb_model_file)
print(f"✅ XGBoost model saved: {xgb_model_file}")

# -----------------------------
# 3. TRAIN BERT CLASSIFIER
# -----------------------------
bert_model_path = os.path.join(BASE_DIR, "models/bert_classifier")
os.makedirs(bert_model_path, exist_ok=True)

# Encode labels
le = LabelEncoder()
ip_data['label_enc'] = le.fit_transform(ip_data['classification'])

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokens = tokenizer(list(ip_data['title'].values),
                   padding=True,
                   truncation=True,
                   return_tensors="pt")

class IPDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {**{key: val[idx] for key, val in self.encodings.items()},
                'labels': self.labels[idx]}

dataset = IPDataset(tokens, ip_data['label_enc'].values)

train_size = int(0.8 * len(dataset))
train_dataset = torch.utils.data.Subset(dataset, range(train_size))
test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(le.classes_)
)

# -----------------------------
# 4. TRAINING ARGUMENTS
# -----------------------------
training_args = TrainingArguments(
    output_dir=bert_model_path,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    logging_dir='./logs',
    logging_strategy="steps",
    logging_steps=5,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()
trainer.save_model(bert_model_path)
tokenizer.save_pretrained(bert_model_path)

label_file = os.path.join(bert_model_path, "label_encoder.pkl")

with open(label_file, "wb") as f:
    pickle.dump(le, f)

print(f"✅ BERT classifier saved in: {bert_model_path}")

print("🎉 All models trained from ip_monitoring database and ready!")