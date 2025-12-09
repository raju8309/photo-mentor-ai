from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import base64
import cv2
import numpy as np

from .services.analysis_service import analyze_frame

app = FastAPI()

# ---- CORS: allow localhost + your Vercel app + Vercel previews ----
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://photo-mentor-ai-six.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    # allow all preview branches like https://photo-mentor-ai-six-xxxxx.vercel.app
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Health / root ----
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "PhotoMentorAI backend running",
        "message": "Backend is healthy!",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "PhotoMentorAI backend running",
        "message": "Backend is healthy!",
    }


# ---- Frame analysis endpoint ----
@app.post("/analyze_frame")
def analyze(payload: dict = Body(...)):
    """
    Accepts JSON with:
      { "image_base64": "data:image/jpeg;base64,..." }
    """
    try:
        image_b64 = payload.get("image_base64") or payload.get("image")
        if not image_b64:
            raise ValueError("Missing 'image_base64' or 'image' in payload")

        # Strip header "data:image/jpeg;base64,..." if present
        b64 = image_b64.split(",")[-1]
        img_bytes = base64.b64decode(b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        result = analyze_frame(frame)
        return result

    except Exception as e:
        # Safe fallback payload so frontend doesn't crash
        return {
            "brightness": 0.0,
            "exposure_hint": f"Error: {e}",
            "face_hint": "Error processing frame.",
            "composition_hint": "Error processing frame.",
            "num_faces": 0,
            "timing_hint": "Error processing frame.",
            "timing_score": 0,
            "expression_label": "no_face",
            "expression_confidence": 0,
            "scene_type": "Unknown",
            "recommended_aperture": "2.8",
            "recommended_shutter": "1/250",
            "recommended_iso": 400,
            "recommended_ev": "+0.3",
        }