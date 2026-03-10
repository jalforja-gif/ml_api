# serve.py
from waitress import serve
from ml_api_app import app  # iyong Flask app

print("Models loaded. Starting server...")  # <- dagdag na ito

serve(app, host="0.0.0.0", port=5000)