import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

from app.models.emotion_model import predict_emotion

# ----------------- Load cascades -------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)

if face_cascade.empty():
    print("⚠️ Warning: Could not load face cascade.")
if eye_cascade.empty():
    print("⚠️ Warning: Could not load eye cascade.")


# ----------------- Face helpers -------------------------


def detect_faces_robust(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    More robust face detector:
    1) Try on resized image with normal settings
    2) If nothing found, retry with looser parameters
    """
    h, w = gray.shape[:2]

    max_side = max(w, h)
    scale = 1.0
    if max_side > 640:
        scale = 640.0 / max_side
        new_size = (int(w * scale), int(h * scale))
        resized = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
    else:
        resized = gray

    faces_small = face_cascade.detectMultiScale(
        resized,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    faces: List[Tuple[int, int, int, int]] = []
    for (x, y, fw, fh) in faces_small:
        x_o = int(x / scale)
        y_o = int(y / scale)
        fw_o = int(fw / scale)
        fh_o = int(fh / scale)
        faces.append((x_o, y_o, fw_o, fh_o))

    # Fallback if nothing detected
    if len(faces) == 0:
        faces_fallback = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30),
        )
        for (x, y, fw, fh) in faces_fallback:
            faces.append((x, y, fw, fh))

    return faces


# ----------------- Main analysis -------------------------


def analyze_frame(frame: np.ndarray) -> Dict[str, Any]:
    """
    Analyze a single BGR frame from the camera.

    Returns:
      - brightness
      - exposure_hint
      - face_hint
      - composition_hint
      - num_faces
      - timing_hint
      - timing_score  (0–100)
      - expression_label        (happy / neutral / serious / eyes_closed / no_face / unknown)
      - expression_confidence   (0–100)
      - faces: list of per-face dicts with emotion for multi-person scenes
    """
    try:
        if frame is None:
            return {
                "brightness": 0.0,
                "exposure_hint": "No frame received.",
                "face_hint": "No frame received.",
                "composition_hint": "No frame received.",
                "num_faces": 0,
                "timing_hint": "No frame received.",
                "timing_score": 0,
                "expression_label": "no_face",
                "expression_confidence": 0,
                "faces": [],
            }

        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ----- Brightness / exposure -----
        mean_brightness = float(gray.mean())

        if mean_brightness < 60:
            exposure_hint = "Too dark – increase ISO or move to better light."
        elif mean_brightness > 190:
            exposure_hint = "Too bright – reduce exposure or move to shade."
        else:
            exposure_hint = "Exposure looks good."

        # ----- Face detection (multi-face) -----
        raw_faces = detect_faces_robust(gray)
        num_faces = int(len(raw_faces))

        face_hint = ""
        composition_hint = ""
        timing_hint = "Waiting for subject…"
        timing_score = 35

        # defaults for global expression
        global_expression_label = "no_face"
        global_expression_confidence = 0

        # list for per-face summary (for future UI)
        faces_summary: List[Dict[str, Any]] = []

        if num_faces == 0:
            face_hint = "No face detected – point the camera at your subject."
            composition_hint = "Try bringing your subject closer to the center."
            timing_hint = "No subject yet – frame your subject before capturing."
            timing_score = 30
        else:
            # choose main subject: biggest face
            main_idx = 0
            max_area = 0
            for i, (x, y, fw, fh) in enumerate(raw_faces):
                area = fw * fh
                if area > max_area:
                    max_area = area
                    main_idx = i

            # process each face safely
            for idx, (x, y, fw, fh) in enumerate(raw_faces):
                # Skip invalid crops
                if fw <= 10 or fh <= 10:
                    continue
                if (
                    x < 0
                    or y < 0
                    or x + fw > gray.shape[1]
                    or y + fh > gray.shape[0]
                ):
                    continue

                try:
                    label, conf = predict_emotion(gray, (x, y, fw, fh))
                except Exception as e:
                    print("[emotion_model] Error:", e)
                    label, conf = "unknown", 0

                faces_summary.append(
                    {
                        "id": idx,
                        "x": int(x),
                        "y": int(y),
                        "w": int(fw),
                        "h": int(fh),
                        "label": label,
                        "confidence": int(conf),
                    }
                )

            # if nothing valid after safety checks
            if not faces_summary:
                face_hint = "Face region unclear – try moving closer to the camera."
                composition_hint = "Try bringing your subject closer to the center."
                timing_hint = "Hold steady and try again."
                timing_score = 40
            else:
                # main subject info based on main_idx
                main_face = next(
                    (f for f in faces_summary if f["id"] == main_idx),
                    faces_summary[0],
                )
                x = main_face["x"]
                y = main_face["y"]
                fw = main_face["w"]
                fh = main_face["h"]

                cx = x + fw / 2
                cy = y + fh / 2

                # Center box
                cx_min, cx_max = w * 0.3, w * 0.7
                cy_min, cy_max = h * 0.3, h * 0.7

                if cx_min < cx < cx_max and cy_min < cy < cy_max:
                    face_hint = "Face nicely centered – focus looks good."
                else:
                    face_hint = "Reframe slightly to bring the face closer to center."

                if cy < h * 0.25:
                    composition_hint = (
                        "Lower the frame a bit to reduce empty space on top."
                    )
                elif cy > h * 0.75:
                    composition_hint = (
                        "Raise the frame – too much space above the subject."
                    )
                else:
                    composition_hint = "Composition looks roughly balanced."

                # ----- Global expression from main face -----
                global_expression_label = main_face["label"]
                global_expression_confidence = main_face["confidence"]

                # ----- Timing logic based on expression -----
                if global_expression_label == "happy":
                    timing_hint = "Great expression – capture now!"
                    timing_score = 92
                elif global_expression_label in ("neutral", "serious"):
                    timing_hint = "Expression is neutral – try getting a smile."
                    timing_score = 72
                elif global_expression_label == "eyes_closed":
                    timing_hint = (
                        "Eyes may be closed or obscured – wait before shooting."
                    )
                    timing_score = 45
                elif global_expression_label == "no_face":
                    timing_hint = "No clear face visible – adjust your framing."
                    timing_score = 35
                else:
                    timing_hint = "Hold steady and watch for a better expression."
                    timing_score = 60

        return {
            "brightness": mean_brightness,
            "exposure_hint": exposure_hint,
            "face_hint": face_hint,
            "composition_hint": composition_hint,
            "num_faces": num_faces,
            "timing_hint": timing_hint,
            "timing_score": int(timing_score),
            "expression_label": global_expression_label,
            "expression_confidence": int(global_expression_confidence),
            "faces": faces_summary,
        }

    except Exception as e:
        # hard safety net so frontend never breaks
        print("[analyze_frame] ERROR:", e)
        return {
            "brightness": 0.0,
            "exposure_hint": "Error processing frame.",
            "face_hint": "Error processing frame.",
            "composition_hint": "Error processing frame.",
            "num_faces": 0,
            "timing_hint": "Error processing frame.",
            "timing_score": 0,
            "expression_label": "no_face",
            "expression_confidence": 0,
            "faces": [],
        }