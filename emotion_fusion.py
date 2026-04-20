from typing import Optional, Tuple

# Emotion reliability weights - face emotion is generally less reliable than text
# unless the camera image is high quality
TEXT_WEIGHT = 0.55
FACE_WEIGHT = 0.45

# Agreement bonus: if both sources agree, boost confidence
AGREEMENT_BONUS = 8

# Emotion priority for conflict resolution
PRIORITY_MAP = {
    "angry": 5,
    "stressed": 4,
    "sad": 3,
    "happy": 2,
    "neutral": 1
}


def fuse_emotions(
    text_result: Optional[Tuple[str, int]],
    face_result: Optional[Tuple[str, int]]
) -> Tuple[str, int]:
    """
    Fuse text and face emotion results using weighted confidence scoring.
    
    Strategy:
    - If both agree → boost confidence
    - If they disagree → use weighted confidence + priority map
    - Single source → return that source
    
    Returns (emotion_label, final_confidence)
    """
    if text_result is None and face_result is None:
        return "neutral", 50

    if text_result is None:
        return face_result

    if face_result is None:
        return text_result

    text_emotion, text_conf = text_result
    face_emotion, face_conf = face_result

    # Case 1: Both sources agree
    if text_emotion == face_emotion:
        # Average confidence + bonus for agreement
        avg_conf = int((text_conf * TEXT_WEIGHT) + (face_conf * FACE_WEIGHT))
        boosted = min(97, avg_conf + AGREEMENT_BONUS)
        return text_emotion, boosted

    # Case 2: Disagreement — use weighted scoring
    text_score = text_conf * TEXT_WEIGHT
    face_score = face_conf * FACE_WEIGHT

    if text_score >= face_score:
        winner_emotion = text_emotion
        winner_conf = int(text_score + (face_score * 0.2))  # partial boost from other
    else:
        winner_emotion = face_emotion
        winner_conf = int(face_score + (text_score * 0.2))

    # Clamp to realistic range
    winner_conf = min(90, max(55, winner_conf))

    return winner_emotion, winner_conf


def fusion_report(
    text_result: Optional[Tuple[str, int]],
    face_result: Optional[Tuple[str, int]]
) -> dict:
    """
    Return a detailed fusion report for display in the UI.
    """
    final_emotion, final_conf = fuse_emotions(text_result, face_result)

    agreement = False
    if text_result and face_result:
        agreement = text_result[0] == face_result[0]

    return {
        "final_emotion": final_emotion,
        "final_confidence": final_conf,
        "text_input": text_result,
        "face_input": face_result,
        "sources_agree": agreement,
        "method": "agreement_boost" if agreement else "weighted_fusion",
        "text_weight": TEXT_WEIGHT,
        "face_weight": FACE_WEIGHT
    }