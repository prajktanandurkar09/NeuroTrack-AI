import re
from collections import defaultdict

# Keyword dictionaries with weights
EMOTION_KEYWORDS = {
    "happy": {
        "keywords": [
            "happy", "joy", "joyful", "excited", "great", "awesome", "wonderful",
            "good", "fantastic", "love", "amazing", "cheerful", "glad", "elated",
            "pleased", "content", "delighted", "thrilled", "fun", "enjoy", "smile",
            "laugh", "positive", "bright", "energetic", "motivated", "confident"
        ],
        "weight": 1.0
    },
    "sad": {
        "keywords": [
            "sad", "unhappy", "depressed", "down", "miserable", "heartbroken",
            "cry", "crying", "tears", "lonely", "hopeless", "lost", "grief",
            "pain", "hurt", "sorrow", "gloomy", "blue", "upset", "broken",
            "disappointed", "low", "empty", "numb", "helpless"
        ],
        "weight": 1.0
    },
    "stressed": {
        "keywords": [
            "stress", "stressed", "tired", "exhausted", "overwhelmed", "burnout",
            "anxious", "anxiety", "nervous", "worry", "worried", "pressure",
            "deadline", "overloaded", "busy", "cannot focus", "can't focus",
            "no energy", "fatigue", "drained", "tense", "panic", "fear", "scared"
        ],
        "weight": 1.0
    },
    "angry": {
        "keywords": [
            "angry", "anger", "furious", "mad", "rage", "hate", "annoyed",
            "frustrated", "irritated", "aggressive", "outrage", "disgusted",
            "bitter", "hostile", "violent", "yell", "scream", "worst", "terrible",
            "horrible", "unfair", "injustice"
        ],
        "weight": 1.0
    },
    "neutral": {
        "keywords": [
            "okay", "ok", "fine", "alright", "normal", "nothing", "usual",
            "average", "meh", "so-so", "regular", "ordinary"
        ],
        "weight": 0.6
    }
}

NEGATIONS = {"not", "no", "never", "don't", "doesn't", "didn't", "won't", "cannot", "can't", "hardly", "barely"}

def preprocess_text(text: str) -> list:
    """Tokenize and clean text."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    tokens = text.split()
    return tokens

def predict_emotion(text: str) -> tuple:
    """
    Predict emotion from text using keyword scoring with negation handling.
    Returns (emotion_label, confidence_percent)
    """
    if not text or not text.strip():
        return "neutral", 50

    tokens = preprocess_text(text)
    scores = defaultdict(float)
    
    negation_active = False
    negation_window = 0

    for i, token in enumerate(tokens):
        # Track negations
        if token in NEGATIONS:
            negation_active = True
            negation_window = 3  # affect next 3 words
        
        if negation_active and negation_window > 0:
            negation_window -= 1
            if negation_window == 0:
                negation_active = False

        for emotion, data in EMOTION_KEYWORDS.items():
            if token in data["keywords"]:
                score = data["weight"]
                if negation_active:
                    # Flip emotion on negation (happy -> sad, sad -> neutral, etc.)
                    if emotion == "happy":
                        scores["sad"] += score * 0.7
                    elif emotion == "sad":
                        scores["neutral"] += score * 0.7
                    elif emotion == "stressed":
                        scores["neutral"] += score * 0.5
                    elif emotion == "angry":
                        scores["neutral"] += score * 0.5
                else:
                    scores[emotion] += score

        # Check bigrams for multi-word phrases
        if i < len(tokens) - 1:
            bigram = f"{token} {tokens[i+1]}"
            for emotion, data in EMOTION_KEYWORDS.items():
                if bigram in data["keywords"]:
                    scores[emotion] += data["weight"] * 1.5  # bigrams are stronger signals

    if not scores:
        return "neutral", 65

    # Normalize scores to confidence
    top_emotion = max(scores, key=scores.get)
    total = sum(scores.values())
    raw_conf = scores[top_emotion] / total if total > 0 else 0

    # Scale confidence to realistic range (55-95%)
    confidence = int(55 + raw_conf * 40)
    confidence = min(95, max(55, confidence))

    return top_emotion, confidence


def get_all_scores(text: str) -> dict:
    """Return scores for all emotions (useful for charts)."""
    if not text or not text.strip():
        return {"neutral": 65}

    tokens = preprocess_text(text)
    scores = defaultdict(float)

    for token in tokens:
        for emotion, data in EMOTION_KEYWORDS.items():
            if token in data["keywords"]:
                scores[emotion] += data["weight"]

    if not scores:
        scores["neutral"] = 1.0

    # Normalize to percentages
    total = sum(scores.values())
    return {e: round((s / total) * 100, 1) for e, s in scores.items()}