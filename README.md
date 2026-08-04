# AnomalyShield — Harmful Speech Detection Chatbot

A local Flask chatbot that blocks harmful English, Bangla, and code-mixed messages, assigns **mild / moderate / severe** severity, records violations, and applies escalating cooldowns. It gives users a constructive alternative instead of silently rejecting them.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Moderation policy

| Finding | Result |
| --- | --- |
| Safe | Message delivered |
| Mild offensive/bullying language | Warning and alternative wording |
| Moderate hate/abuse | Warning and alternative wording |
| Severe threat/violence | Blocked for 5 minutes immediately |
| 3 violations | 1 minute block |
| 5+ violations | 5 minute block |

The demo stores its state in memory so restarting the server resets all counts. Production deployments should use authenticated accounts, a database or Redis, audit logging, appeals/review, rate limiting, and a properly evaluated trained classifier.

## Dataset-backed model (optional)

The supplied source files were extracted to `data/raw/` (ignored by Git). To train a character n-gram multilingual baseline:

```powershell
python train_model.py --data-dir data/raw
```

Use `--max-rows 20000` for a quicker experiment. Evaluate the model on held-out English and Bangla data before moderation use; keyword/rule signals in the running demo remain intentionally visible and easy to audit.
