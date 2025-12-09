import os
import cv2
import numpy as np
from typing import Tuple, Optional

# Path to the ONNX model sitting in THIS folder:
# backend/app/models/emotion-ferplus-8.onnx
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "emotion-ferplus-8.onnx")
)

# FER+ defines 8 emotion classes (order must match the model output)
FERPLUS_LABELS = [
    "neutral",
    "happy",
    "surprise",
    "sad",
    "anger",
    "disgust",
    "fear",
    "contempt",
]

# Map FER+ labels -> app-level labels
APP_LABEL_MAPPING = {
    "happy": "happy",
    "neutral": "neutral",
    "surprise": "neutral",  # treat surprise as neutral/good
    "sad": "serious",
    "anger": "serious",
    "disgust": "serious",
    "fear": "serious",
    "contempt": "serious",
}

_net = None
_load_failed = False


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _load_net():
    """Load the ONNX model once and cache it."""
    global _net, _load_failed

    if _net is not None:
        return _net
    if _load_failed:
        return None

    if not os.path.exists(MODEL_PATH):
        print(f"[emotion_model] ONNX model not found at: {MODEL_PATH}")
        _load_failed = True
        return None

    try:
        _net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        print("[emotion_model] Loaded FER+ ONNX model.")
    except Exception as e:
        print("[emotion_model] Failed to load FER+ model:", e)
        _load_failed = True
        _net = None

    return _net


def _crop_face_square(
    gray_frame: np.ndarray, face_box: Tuple[int, int, int, int]
) -> Optional[np.ndarray]:
    """
    Take the original grayscale frame + (x, y, w, h) and return
    a slightly padded *square* crop around the face.
    """
    x, y, w, h = face_box
    h_img, w_img = gray_frame.shape[:2]

    cx = x + w // 2
    cy = y + h // 2
    side = int(max(w, h) * 1.1)  # pad 10%

    x1 = max(cx - side // 2, 0)
    y1 = max(cy - side // 2, 0)
    x2 = min(cx + side // 2, w_img)
    y2 = min(cy + side // 2, h_img)

    if x2 <= x1 or y2 <= y1:
        return None

    return gray_frame[y1:y2, x1:x2]


def _predict_from_crop(face_gray: np.ndarray):
    """
    Low-level helper: run the FER+ model on a *face crop* only.

    Returns:
      raw_label: one of FERPLUS_LABELS
      conf: float in [0, 1]
      probs_dict: {label: prob}
    """
    net = _load_net()
    if net is None:
        # Model missing or failed to load
        return "missing_model", 0.0, {}

    if face_gray is None or face_gray.size == 0:
        return "unknown", 0.0, {}

    # Ensure grayscale
    if len(face_gray.shape) == 3:
        face_gray = cv2.cvtColor(face_gray, cv2.COLOR_BGR2GRAY)

    # FER+ expects 64x64 grayscale, N x 1 x 64 x 64
    try:
        resized = cv2.resize(face_gray, (64, 64))
    except Exception as e:
        print("[emotion_model] Resize failed:", e)
        return "unknown", 0.0, {}

    blob = resized.astype("float32")
    blob = blob[np.newaxis, np.newaxis, :, :]  # (1,1,64,64)

    # Forward pass
    net.setInput(blob)
    out = net.forward()  # shape (1, 8)
    out = out[0]

    probs = _softmax(out)
    idx = int(np.argmax(probs))
    conf = float(probs[idx])

    label = FERPLUS_LABELS[idx]
    probs_dict = {
        FERPLUS_LABELS[i]: float(probs[i]) for i in range(len(FERPLUS_LABELS))
    }

    return label, conf, probs_dict


def predict_emotion(
    gray_or_face: np.ndarray, face_box: Optional[Tuple[int, int, int, int]] = None
):
    """
    High-level API used by the backend.

    You can call this in **two ways**:

    1) New way (recommended – lets us do a smart crop):
         app_label, conf_pct = predict_emotion(gray_frame, (x, y, w, h))

    2) Backwards-compatible way (already-cropped face):
         app_label, conf_pct = predict_emotion(face_crop)

    Returns:
      app_label: one of {"happy", "neutral", "serious", "unknown", "no_face"}
      conf_pct: int 0–100
    """
    # Decide which image to feed into the model
    if face_box is None:
        # Assume gray_or_face is already a cropped face
        face_crop = gray_or_face
    else:
        # gray_or_face is the full grayscale frame
        face_crop = _crop_face_square(gray_or_face, face_box)

    if face_crop is None or face_crop.size == 0:
        return "no_face", 0

    raw_label, conf, _ = _predict_from_crop(face_crop)

    # If model missing or failed, keep classic hints but mark unknown
    if raw_label in ("missing_model", "unknown"):
        return "unknown", 0

    # Map FER+ label -> our app label
    app_label = APP_LABEL_MAPPING.get(raw_label, "unknown")

    # Convert to percentage
    conf_pct = int(conf * 100.0)

    # If confidence is very low, don't trust the label
    if conf_pct < 35:
        return "unknown", conf_pct

    return app_label, conf_pct