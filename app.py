"""AnomalyShield: a small multilingual harmful-speech moderation chatbot."""
from __future__ import annotations

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

SEVERITY_LABELS = {
    "safe": "Safe message",
    "mild": "Model-detected offensive language",
    "moderate": "Model-detected hate / abuse",
    "severe": "Model-detected severe toxicity / threat",
}


def client_id() -> str:
    """Use a supplied demo id so multiple browser tabs share a moderation record."""
    return request.headers.get("X-User-Id") or request.remote_addr or "anonymous"


def model_assessment(text: str) -> tuple[str, str]:
    """Classify with the trained dataset model only—no keywords or heuristics."""
    probabilities = MODEL.predict_proba([text])[0]
    classes = MODEL.named_steps["classifier"].classes_
    index = int(probabilities.argmax())
    severity = str(classes[index])
    confidence = float(probabilities[index])
    return severity, f"{SEVERITY_LABELS[severity]} ({confidence:.0%} confidence)"


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
    if MODEL is None:
        return jsonify(error="No trained model found. Run: python train_model.py --data-dir data/raw, then restart the app."), 503

    user = users[client_id()]
    now = time.time()
    seconds_left = max(0, round(user["blocked_until"] - now))
    if seconds_left:
        return jsonify(blocked=True, seconds_left=seconds_left, violations=user["violations"], allowed=False)

    severity, category = model_assessment(text)
    unsafe = severity != "safe"
    cooldown = 0
    if unsafe:
        user["violations"] += 1
        # Severe incidents block immediately; repeated lower-level incidents escalate.
        if severity == "severe":
            cooldown = 180
        elif user["violations"] >= 5:
            cooldown = 300
        elif user["violations"] >= 3:
            cooldown = 180
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
