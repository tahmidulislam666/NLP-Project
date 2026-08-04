"""AnomalyShield: a small multilingual harmful-speech moderation chatbot."""
from __future__ import annotations

import re
import time
from collections import defaultdict
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")

# A trained pipeline is optional: the app starts without ML dependencies and uses
# it automatically when train_model.py has produced this artifact.
MODEL = None
try:
    import joblib
    model_path = Path("models/toxicity_pipeline.joblib")
    if model_path.exists():
        MODEL = joblib.load(model_path)
except (ImportError, OSError):
    pass

# Demo-grade in-memory state. Replace this with Redis/database state in production.
users: dict[str, dict] = defaultdict(lambda: {"violations": 0, "blocked_until": 0.0})

RULES = {
    "severe": {
        "label": "Threat / violence",
        "patterns": [r"\b(kill|murder|rape|shoot|stab|bomb)\b", r"\b(i('ll| will)|we('ll| will))\s+(kill|hurt|find)\b", r"মেরে ফেল|খুন কর|ধর্ষণ|গুলি কর|বোমা"],
    },
    "moderate": {
        "label": "Hate / abusive language",
        "patterns": [r"\b(hate you|go die|terrorist|nazi)\b", r"\b(fuck|bitch|asshole|bastard)\b", r"হারামি|শালা|কুত্তা|গালি|ঘৃণা করি"],
    },
    "mild": {
        "label": "Offensive / bullying language",
        "patterns": [r"\b(idiot|stupid|dumb|loser|shut up)\b", r"\b(you suck|worthless)\b", r"বোকা|পাগল|নির্বোধ|অপদার্থ"],
    },
}
SEVERITY_SCORE = {"safe": 0, "mild": 1, "moderate": 2, "severe": 3}


def client_id() -> str:
    """Use a supplied demo id so multiple browser tabs share a moderation record."""
    return request.headers.get("X-User-Id") or request.remote_addr or "anonymous"


def rule_assessment(text: str) -> tuple[str, str]:
    text = text.lower()
    for severity in ("severe", "moderate", "mild"):
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in RULES[severity]["patterns"]):
            return severity, RULES[severity]["label"]
    # Excessive shouting is a low-confidence bullying signal.
    letters = [c for c in text if c.isalpha()]
    if len(letters) > 10 and sum(c.isupper() for c in letters) / len(letters) > 0.8:
        return "mild", "Aggressive tone"
    if MODEL is not None:
        probability = float(MODEL.predict_proba([text])[0][1])
        if probability >= 0.80:
            return "moderate", f"Model-detected toxicity ({probability:.0%} confidence)"
    return "safe", "Safe message"


def safer_alternative(text: str, severity: str) -> str:
    if severity == "severe":
        return "I am very upset. I need space, and I want to resolve this safely."
    if severity == "moderate":
        return "I strongly disagree with this, but I want to explain my concern respectfully."
    if severity == "mild":
        return "I disagree. Could you explain your point more clearly?"
    return text


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.post("/api/messages")
def moderate_message():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("message", "")).strip()
    if not text:
        return jsonify(error="Please type a message."), 400
    if len(text) > 1000:
        return jsonify(error="Messages must be 1,000 characters or fewer."), 400

    user = users[client_id()]
    now = time.time()
    seconds_left = max(0, round(user["blocked_until"] - now))
    if seconds_left:
        return jsonify(blocked=True, seconds_left=seconds_left, violations=user["violations"], allowed=False)

    severity, category = rule_assessment(text)
    unsafe = severity != "safe"
    cooldown = 0
    if unsafe:
        user["violations"] += 1
        # Severe incidents block immediately; repeated lower-level incidents escalate.
        if severity == "severe":
            cooldown = 300
        elif user["violations"] >= 5:
            cooldown = 300
        elif user["violations"] >= 3:
            cooldown = 60
        user["blocked_until"] = now + cooldown

    return jsonify(
        allowed=not unsafe,
        unsafe=unsafe,
        severity=severity,
        category=category,
        violations=user["violations"],
        cooldown=cooldown,
        safer_alternative=safer_alternative(text, severity),
    )


@app.post("/api/reset")
def reset_demo():
    users[client_id()] = {"violations": 0, "blocked_until": 0.0}
    return jsonify(ok=True, violations=0)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
