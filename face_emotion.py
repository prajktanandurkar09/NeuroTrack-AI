import cv2
import numpy as np
from PIL import Image
import io

def predict_face_emotion(image_input) -> tuple:
    """
    Predict emotion from face image using OpenCV analysis.
    Uses brightness, contrast, and edge density as proxy signals.
    Returns (emotion_label, confidence_percent)
    """
    try:
        # Handle Streamlit UploadedFile or PIL Image or bytes
        if hasattr(image_input, "read"):
            img_bytes = image_input.read()
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = np.array(pil_img)
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            img_bytes = image_input.getvalue()
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = np.array(pil_img)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # --- Feature Extraction ---
        avg_brightness = float(np.mean(gray))
        std_brightness = float(np.std(gray))

        # Detect edges (proxy for expression intensity)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / edges.size

        # Detect face region using Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        face_detected = len(faces) > 0

        # --- Scoring Logic ---
        # Bright + high edge density → likely animated/happy expression
        # Dark + low variance → possibly sad/tired
        # Medium brightness + high std → possibly stressed

        if face_detected:
            # Analyze upper face region (forehead/brow area) for tension
            (fx, fy, fw, fh) = faces[0]
            upper_face = gray[fy:fy + int(fh * 0.4), fx:fx + fw]
            upper_std = float(np.std(upper_face)) if upper_face.size > 0 else std_brightness
        else:
            upper_std = std_brightness

        # Decision tree based on extracted features
        if avg_brightness > 160 and edge_density > 0.05:
            emotion, confidence = "happy", _calc_confidence(avg_brightness, 160, 255, 75, 92)

        elif avg_brightness > 130 and edge_density > 0.03 and upper_std < 40:
            emotion, confidence = "neutral", _calc_confidence(avg_brightness, 130, 160, 68, 82)

        elif avg_brightness > 110 and upper_std > 50 and edge_density > 0.04:
            emotion, confidence = "stressed", _calc_confidence(upper_std, 50, 90, 65, 85)

        elif avg_brightness < 90 and edge_density < 0.03:
            emotion, confidence = "sad", _calc_confidence(100 - avg_brightness, 10, 100, 60, 80)

        elif avg_brightness > 100 and edge_density > 0.06 and upper_std > 55:
            emotion, confidence = "angry", _calc_confidence(edge_density, 0.06, 0.15, 62, 82)

        elif avg_brightness > 130:
            emotion, confidence = "neutral", 70

        else:
            emotion, confidence = "sad", 60

        return emotion, confidence

    except Exception as e:
        print(f"[Face Emotion Error]: {e}")
        return "neutral", 50


def _calc_confidence(value: float, low: float, high: float, min_conf: int, max_conf: int) -> int:
    """Map a feature value to a confidence percentage."""
    ratio = (value - low) / (high - low) if (high - low) != 0 else 0.5
    ratio = max(0.0, min(1.0, ratio))
    return int(min_conf + ratio * (max_conf - min_conf))


def get_face_features(image_input) -> dict:
    """Return raw face analysis features (for debug/display)."""
    try:
        if hasattr(image_input, "getvalue"):
            img_bytes = image_input.getvalue()
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = np.array(pil_img)
        else:
            img = np.array(image_input)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        return {
            "brightness": round(float(np.mean(gray)), 2),
            "contrast": round(float(np.std(gray)), 2),
            "edge_density": round(float(np.sum(edges > 0)) / edges.size, 4),
            "faces_detected": len(faces),
            "image_size": img.shape[:2]
        }
    except Exception as e:
        return {"error": str(e)}