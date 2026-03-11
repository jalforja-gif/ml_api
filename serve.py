from waitress import serve
from ml_api_app import app
import os

port = int(os.environ.get("PORT", 5000))

print(f"Starting ML API on 0.0.0.0:{port} ...")
serve(app, host="0.0.0.0", port=port)
