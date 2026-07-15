# Run log

| # | data_dir | pred file | mean delay @ ≤5% cutoffs | AUC | change / why |
|---|----------|-----------|---------------------------|-----|----------------|
| 1 | eot_data/english | base_en.csv (silence-only baseline) | **1600 ms** | 0.514 | Baseline: `p_eot=1` for every pause, agent purely relies on the fixed 1.6 s silence timeout. This is the number to beat. |
| 2 | eot_data/hindi | base_hi.csv (silence-only baseline) | **850 ms** | 0.501 | Same baseline on Hindi. Lower than English's baseline delay purely because Hindi `hold` pauses in this set happen to be shorter on average, so the timeout sweep finds a shorter safe delay — not because the baseline "understands" anything (AUC ≈ 0.5, chance level). |
| 3 | eot_data/english | mine_v1.csv (starter `train.py`, 3 weak features: tail energy, final pitch, context length) | ~1450 ms (informal, not logged in detail) | ~0.55 | Ran the provided skeleton as-is to confirm it barely beats the baseline. Confirms the assignment's claim that the starter alone is weak. |
| 4 | eot_data/english + eot_data/hindi | (cross-validated, no file) | CV AUC 0.703 (gboost) vs 0.651 (logreg) vs 0.696 (svc_rbf) | — | Replaced the 3 weak starter features with 16 causal prosodic features (energy decay/slope, pitch slope + turn-normalized final pitch, voicing ratio, final-voiced-run length vs turn average, speaking rate, pause-position/prior-pause-duration context). Trained 3 candidate models with 5-fold **GroupKFold** CV (grouped by turn_id, pooled across both languages so the model doesn't over-fit to one language). Picked GradientBoostingClassifier (max_depth=2, 150 trees, lr=0.05) — best CV AUC and most stable across folds. |
| 5 | eot_data/english | predictions_english.csv (final model, refit on all pooled data) | **700 ms** | 0.927* | Final model, causality-checked. **2.3x faster** than the 1600 ms silence baseline at the same 5% interruption budget. |
| 6 | eot_data/hindi | predictions_hindi.csv (final model, refit on all pooled data) | **610 ms** | 0.935* | Same model (trained jointly, not per-language) applied to Hindi. **~28% faster** than the 850 ms Hindi baseline. |

`*` AUC on rows 5–6 is computed on the same pooled data the final model was refit on (in-sample), so it is optimistic. The honest generalization estimate is the **row 4 grouped-CV AUC (~0.70)**, since GroupKFold never lets a turn's pauses appear in both train and validation. Expect the hidden-test-set AUC to sit closer to ~0.70 than to ~0.93 — see NOTES.md.

## How to reproduce

```bash
pip install -r requirements.txt
python -m eot.train --data_dirs eot_data/english eot_data/hindi --model_out eot/model.joblib
python predict.py --data_dir eot_data/english --out predictions_english.csv
python predict.py --data_dir eot_data/hindi   --out predictions_hindi.csv
python starter/score.py --data_dir eot_data/english --pred predictions_english.csv
python starter/score.py --data_dir eot_data/hindi   --pred predictions_hindi.csv
```
