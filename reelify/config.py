import os
from pathlib import Path

# Repo root (one level above reelify/)
BASE_DIR = Path(__file__).resolve().parent.parent

STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Outputs
IMG_DIR = STATIC_DIR / "outputs" / "images"
VID_DIR = STATIC_DIR / "outputs" / "videos"
IMG_DIR.mkdir(parents=True, exist_ok=True)
VID_DIR.mkdir(parents=True, exist_ok=True)

# Secrets / API keys (Render reads from environment variables)
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
NEWSAPI_BASE_URL = os.getenv("NEWSAPI_BASE_URL", "https://newsapi.org/v2")

# Feature flags (Render: keep 0)
ENABLE_MEDIA = os.environ.get("ENABLE_MEDIA", "0") == "1"
USE_LLAMA = os.environ.get("USE_LLAMA", "0") == "1"

# Optional admin password override
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "changeme123")
