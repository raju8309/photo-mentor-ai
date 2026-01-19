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


def validate_face_region(gray: np.ndarray, face_box: Tuple[int, int, int, int]) -> bool:
    """
    Additional validation to check if a detected region is likely a face.
    Uses basic heuristics about face regions.
    """
    x, y, w, h = face_box
    
    # Check if region is within image bounds
    h_img, w_img = gray.shape
    if x < 0 or y < 0 or x + w >= w_img or y + h >= h_img:
        return False
    
    # Extract face region
    face_region = gray[y:y+h, x:x+w]
    
    if face_region.size == 0:
        return False
    
    # Check for reasonable variance (faces should have texture)
    variance = face_region.var()
    if variance < 100:  # Too uniform, likely not a face
        return False
    
    # Check edge density (faces should have reasonable edge content)
    edges = cv2.Canny(face_region, 50, 150)
    edge_density = np.sum(edges > 0) / (w * h)
    if edge_density < 0.02:  # Too few edges
        return False
    
    return True


def detect_faces_robust(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Much more robust face detector with multiple strategies:
    1) Histogram equalization for better contrast
    2) Multiple detection attempts with different parameters
    3) Different image scales
    4) Face filtering based on aspect ratio and size
    """
    h, w = gray.shape[:2]
    
    # Apply histogram equalization for better contrast
    gray_eq = cv2.equalizeHist(gray)
    
    all_faces = []
    
    # Strategy 1: Normal detection on equalized image
    for scale_factor in [1.05, 1.1, 1.15]:
        for min_neighbors in [3, 4, 5]:
            for min_size in [(30, 30), (40, 40), (50, 50)]:
                faces = face_cascade.detectMultiScale(
                    gray_eq,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=min_size,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                all_faces.extend(faces)
    
    # Strategy 2: Detection on resized images
    max_side = max(w, h)
    scales_to_try = [0.8, 1.0, 1.2] if max_side > 800 else [0.7, 1.0, 1.3]
    
    for scale in scales_to_try:
        if scale == 1.0:
            gray_scaled = gray_eq
        else:
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            if new_w > 0 and new_h > 0:
                gray_scaled = cv2.resize(gray_eq, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                gray_scaled = gray_eq
        
        faces = face_cascade.detectMultiScale(
            gray_scaled,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(35, 35),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Scale coordinates back to original image size
        if scale != 1.0:
            faces = [(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) for (x, y, w, h) in faces]
        
        all_faces.extend(faces)
    
    # Strategy 3: Very relaxed parameters as last resort
    if len(all_faces) == 0:
        faces_relaxed = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.03,
            minNeighbors=2,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        all_faces.extend(faces_relaxed)
    
    # Filter and merge overlapping faces
    if not all_faces:
        return []
    
    # Remove obvious false positives based on aspect ratio and validate regions
    valid_faces = []
    for (x, y, w, h) in all_faces:
        aspect_ratio = w / h
        # Face aspect ratio is typically between 0.7 and 1.5
        if 0.6 <= aspect_ratio <= 1.6 and w >= 20 and h >= 20:
            # Additional validation
            if validate_face_region(gray, (x, y, w, h)):
                valid_faces.append((x, y, w, h))
    
    # Merge overlapping faces (keep the largest one in each region)
    if not valid_faces:
        return []
    
    merged_faces = []
    for face in valid_faces:
        is_duplicate = False
        for existing in merged_faces:
            # Check if faces overlap significantly
            overlap_ratio = calculate_overlap_ratio(face, existing)
            if overlap_ratio > 0.3:  # 30% overlap threshold
                # Keep the larger face
                if face[2] * face[3] > existing[2] * existing[3]:
                    merged_faces.remove(existing)
                    merged_faces.append(face)
                is_duplicate = True
                break
        if not is_duplicate:
            merged_faces.append(face)
    
    return merged_faces


def calculate_overlap_ratio(face1: Tuple[int, int, int, int], face2: Tuple[int, int, int, int]) -> float:
    """
    Calculate the overlap ratio between two face rectangles.
    """
    x1, y1, w1, h1 = face1
    x2, y2, w2, h2 = face2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


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
        if frame is None or frame.size == 0:
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
        if h <= 0 or w <= 0:
            return {
                "brightness": 0.0,
                "exposure_hint": "Invalid frame dimensions.",
                "face_hint": "Invalid frame dimensions.",
                "composition_hint": "Invalid frame dimensions.",
                "num_faces": 0,
                "timing_hint": "Invalid frame dimensions.",
                "timing_score": 0,
                "expression_label": "no_face",
                "expression_confidence": 0,
                "faces": [],
            }
            
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
        
        # Debug logging
        if num_faces > 0:
            print(f"[face_detection] Found {num_faces} face(s): {raw_faces}")
        else:
            print("[face_detection] No faces detected")

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