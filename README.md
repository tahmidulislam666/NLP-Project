# AnomalyShield — Harmful Speech Detection Chatbot

A local Flask chatbot that uses a model trained on the supplied Jigsaw and Bangla datasets to block harmful English, Bangla, and code-mixed messages. It assigns **mild / moderate / severe** severity, records violations, and applies escalating cooldowns.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py --data-dir data/raw
python app.py
```

Then open `http://127.0.0.1:5000`.

## Moderation policy

| Finding | Result |
| --- | --- |
| Safe | Message delivered |
| Mild or moderate unsafe message | Message blocked, violation recorded, and alternative wording shown |
| Severe unsafe message (fewer than 5 violations) | Message blocked and sending paused for 3 minutes |
| 3–4 violations | Message blocked and sending paused for 3 minutes |
| 5+ violations | 5 minute block |

The app has no rule-based, keyword, or sentiment fallback: it refuses messages until a trained model is available. The demo stores state in memory so restarting the server resets all counts. Production deployments should use authenticated accounts, a database or Redis, audit logging, appeals/review, rate limiting, and a properly evaluated trained classifier.

## Dataset-backed model (required)

The supplied source files were extracted to `data/raw/` (ignored by Git). Train this character n-gram multilingual model before starting the app:

```powershell
python train_model.py --data-dir data/raw
```

Use `--max-rows 20000` for a quicker experiment. Evaluate the model on held-out English and Bangla data before moderation use.
