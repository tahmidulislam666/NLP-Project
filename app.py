"""AnomalyShield: a small multilingual harmful-speech moderation chatbot."""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")

# A trained pipeline is optional: the app starts without ML dependencies and uses
# it automatically when train_model.py has produced this artifact.
MODEL_BUNDLE = None
MODEL_ERROR = None
try:
    import joblib
    model_path = Path("models/toxicity_pipeline.joblib")
    if model_path.exists():
        candidate = joblib.load(model_path)
        if isinstance(candidate, dict) and {"safety_model", "severity_model"}.issubset(candidate):
            MODEL_BUNDLE = candidate
        else:
            MODEL_ERROR = "The saved model uses the old single-stage format. Retrain it with train_model.py."
except (ImportError, OSError):
    MODEL_ERROR = "The trained model could not be loaded."

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
    safety_model = MODEL_BUNDLE["safety_model"]
    safety_probabilities = safety_model.predict_proba([text])[0]
    safety_classes = list(safety_model.named_steps["classifier"].classes_)
    unsafe_probability = float(safety_probabilities[safety_classes.index("unsafe")])
    if unsafe_probability < 0.5:
        safe_probability = float(safety_probabilities[safety_classes.index("safe")])
        return "safe", f"{SEVERITY_LABELS['safe']} ({safe_probability:.0%} confidence)"
    severity_model = MODEL_BUNDLE["severity_model"]
    severity_probabilities = severity_model.predict_proba([text])[0]
    severity_classes = list(severity_model.named_steps["classifier"].classes_)
    severity = str(severity_classes[int(severity_probabilities.argmax())])
    confidence = float(severity_probabilities.max())
    return severity, f"{SEVERITY_LABELS[severity]} ({confidence:.0%} severity confidence)"


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
    if MODEL_BUNDLE is None:
        detail = MODEL_ERROR or "No trained model found."
        return jsonify(error=f"{detail} Run: python train_model.py --data-dir data/raw, then restart the app."), 503

    user = users[client_id()]
    now = time.time()
    seconds_left = max(0, round(user["blocked_until"] - now))
    if seconds_left:
        return jsonify(blocked=True, seconds_left=seconds_left, violations=user["violations"], allowed=False)

    severity, category = model_assessment(text)
    unsafe = severity != "safe"
    cooldown = 0
    violation_recorded = False
    if unsafe:
        # Mild content is blocked and rewritten, but does not penalize the user.
        # Only moderate and severe messages add to the escalation record.
        if severity in {"moderate", "severe"}:
            user["violations"] += 1
            violation_recorded = True
            # Five violations always receive the longest cooldown, regardless of severity.
            if user["violations"] >= 5:
                cooldown = 300
            elif severity == "severe":
                cooldown = 180
            elif user["violations"] >= 3:
                cooldown = 180
        user["blocked_until"] = now + cooldown

    return jsonify(
        allowed=not unsafe,
        unsafe=unsafe,
        severity=severity,
        category=category,
        violations=user["violations"],
        violation_recorded=violation_recorded,
        cooldown=cooldown,
        safer_alternative=safer_alternative(text, severity),
    )


@app.post("/api/reset")
def reset_demo():
    users[client_id()] = {"violations": 0, "blocked_until": 0.0}
    return jsonify(ok=True, violations=0)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
