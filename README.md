# End-of-Turn (EOT) Detection

Predicts, for every silence pause in a user's turn, `p_eot` — the probability the user
is actually done talking — from causal audio features only (no pretrained models,
no ASR, CPU only). Built for the Plivo EOT take-home assignment.

**Result:** mean response delay at ≤5% interrupted turns drops from the
silence-only baseline (1600 ms English / 850 ms Hindi) to **700 ms English /
610 ms Hindi**. Full writeup: [`SUMMARY.html`](SUMMARY.html). Run history:
[`RUNLOG.md`](RUNLOG.md). Signal/failure notes: [`NOTES.md`](NOTES.md).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt

# (re)train (optional — a trained model is already committed at eot/model.joblib)
python -m eot.train --data_dirs eot_data/english eot_data/hindi --model_out eot/model.joblib

# predict (the required deliverable interface)
python predict.py --data_dir eot_data/english --out predictions_english.csv
python predict.py --data_dir eot_data/hindi   --out predictions_hindi.csv

# score (official scorer, provided in starter/)
python starter/score.py --data_dir eot_data/english --pred predictions_english.csv
python starter/score.py --data_dir eot_data/hindi   --pred predictions_hindi.csv
```

`predict.py` works on any folder with the same `audio/` + `labels.csv` schema,
including data it has never seen — it never reads the `label` column or the
current pause's `pause_end`.

## Repo layout

```
predict.py            required CLI entry point
eot/features.py        causal feature extraction (read this to verify causality)
eot/train.py            training + grouped cross-validation + model selection
eot/model.joblib         trained model (StandardScaler/GradientBoosting pipeline)
predictions_english.csv  required deliverable
predictions_hindi.csv    required deliverable
RUNLOG.md, NOTES.md, SUMMARY.html   required deliverables
starter/                given starter kit (baseline.py, score.py, features.py, train.py)
eot_data/               given dev data (english/, hindi/) — see note below
requirements.txt
```

## A note on `eot_data/`

The dev-data folder (~68 MB of wav files) is included here for completeness/reproducibility,
but many teams prefer **not** to commit large binary audio to a git repo. If you'd rather keep
the repo small:

```bash
git rm -r --cached eot_data
echo "eot_data/" >> .gitignore
```

and instead document how to fetch/unzip `eot_handout.zip` into `eot_data/` before running the
commands above. `predict.py`/`eot/train.py` only need the folder to exist locally — nothing
about the code assumes it's committed to git.

## Rules followed

- Laptop CPU only, no GPU/cloud training.
- Libraries used: numpy, scipy, scikit-learn, pandas, soundfile, joblib (matplotlib for the one
  chart in `SUMMARY.html`). No pretrained weights, no external datasets, no ASR/TTS APIs.
- Causality: every feature in `eot/features.py` is computed from frames whose time range ends
  at or before `pause_start`; `pause_end` of the current pause is never read as a feature.
