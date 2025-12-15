import os

def _split_csv(value: str):
    return [v.strip() for v in value.split(",") if v.strip()]

# Comma-separated list of allowed origins for CORS
# Example:
# ALLOWED_ORIGINS="http://localhost:3000,https://photo-mentor-ai-six.vercel.app"
ALLOWED_ORIGINS = _split_csv(
    os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
)

# Optional: allow configuring frontend URL separately if you want later
FRONTEND_URL = os.getenv("FRONTEND_URL", "")