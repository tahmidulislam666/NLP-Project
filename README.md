# AnomalyShield — Harmful Speech Detection Chatbot

A local Flask chatbot that uses a two-stage model trained on the supplied Jigsaw and Bangla datasets. It first identifies **safe / unsafe** messages, then assigns unsafe messages **mild / moderate / severe** severity before applying violations and cooldowns.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py --data-dir data/raw
python app.py
```

Then open `http://127.0.0.1:5000`. Retrain whenever `train_model.py` changes; the app requires the current two-stage model format.

## Moderation policy

| Finding | Result |
| --- | --- |
| Safe | Message delivered |
| Mild unsafe message | Message blocked and alternative wording shown; no violation recorded |
| Moderate unsafe message | Message blocked, violation recorded, and alternative wording shown |
| Severe unsafe message (fewer than 5 violations) | Message blocked and sending paused for 3 minutes |
| 3–4 violations | Message blocked and sending paused for 3 minutes |
| 5+ violations | 5 minute block |

The app has no rule-based, keyword, or sentiment fallback: it refuses messages until a trained model is available. The demo stores state in memory so restarting the server resets all counts. Production deployments should use authenticated accounts, a database or Redis, audit logging, appeals/review, rate limiting, and a properly evaluated trained classifier.

## Dataset-backed model (required)

The supplied source files were extracted to `data/raw/` (ignored by Git). Train this character n-gram multilingual model before starting the app:

```powershell
python train_model.py --data-dir data/raw
```

When `data/raw/test.csv` and `data/raw/test_labels.csv` are present, the script trains on Jigsaw training rows plus 80% of Bangla rows, then evaluates against Jigsaw's independent official test rows with valid labels (rows containing `-1` are excluded) and the unseen 20% Bangla hold-out. If those files are unavailable, it falls back to a stratified combined split while still reporting the separate Bangla hold-out.

It writes two detailed reports after training:

- `models/stage1_evaluation_metrics.json` evaluates the shared **safe / unsafe** Stage 1 separately on English Jigsaw test data and the Bangla hold-out.
- `models/stage2_evaluation_metrics.json` evaluates English **mild / moderate / severe** Stage 2 using only known-unsafe English test messages.

Both reports contain accuracy, weighted and macro F1, ROC-AUC, and per-class results. Use the Stage 1 report for multilingual safety performance and the Stage 2 report for English severity performance.

Use `--max-rows 20000` for a quicker experiment. Report these held-out metrics—not training-set results—when presenting the model.
