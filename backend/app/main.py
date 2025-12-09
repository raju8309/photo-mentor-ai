from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import base64
import cv2
import numpy as np

from .services.analysis_service import analyze_frame

# You can edit title/description later if you want
app = FastAPI(
    title="PhotoMentor AI Backend",
    description="AI-powered real-time photography mentor backend",
    version="1.0.0",
)

# ----------------------------------------------------
# CORS: allow local dev + any HTTPS frontend (Render)
# ----------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    # For now allow all origins so Render static site can call this.
    # Later you can restrict to your frontend URL, e.g.:
    # allow_origins=["https://photo-mentor-frontend.onrender.com"]
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------
# Root / health endpoints
# ----------------------------------------------------
@app.get("/")
def root():
    """
    Simple root route so Render's HEAD / check doesn't return 404.
    """
    return {
        "status": "ok",
        "service": "PhotoMentorAI backend running",
        "message": "Backend is healthy!",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# ----------------------------------------------------
# Main AI endpoint: /analyze_frame
# ----------------------------------------------------
@app.post("/analyze_frame")
def analyze(payload: dict = Body(...)):
    """
    Accepts JSON with:
      { "image_base64": "data:image/jpeg;base64,..." }

    Decodes the image, converts to OpenCV BGR frame,
    then passes it into analyze_frame() in analysis_service.
    """
    try:
        image_b64 = payload.get("image_base64") or payload.get("image")
        if not image_b64:
            raise ValueError("Missing 'image_base64' or 'image' in payload")

        # Strip header "data:image/jpeg;base64," if present
        b64 = image_b64.split(",")[-1]
        img_bytes = base64.b64decode(b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Could not decode image data")

        result = analyze_frame(frame)
        return result

    except Exception as e:
        # Safe fallback so frontend UI keeps working even on error
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