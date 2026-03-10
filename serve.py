from waitress import serve
from ml_api_app import app
import os

print("Models loaded. Starting server...")

# Use the port Render provides, default to 5000 locally
port = int(os.environ.get("PORT", 5000))

serve(app, host="0.0.0.0", port=port)
