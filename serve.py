# serve.py
from waitress import serve
from ml_api_app import app
import os

port = int(os.environ.get("PORT", 5000))  # Use Render's PORT if available
host = "0.0.0.0"

print(f"Models loaded. Starting server on {host}:{port}...")
serve(app, host=host, port=port)
