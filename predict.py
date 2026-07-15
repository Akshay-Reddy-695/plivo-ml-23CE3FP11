"""Required deliverable interface:

    python predict.py --data_dir <folder> --out predictions.csv

Loads the trained model saved in eot/model.joblib and scores every pause
in <folder>/labels.csv, writing predictions.csv with columns
turn_id,pause_index,p_eot.

Works on ANY folder with the same structure/labels schema (audio/ + labels.csv),
including ones the model has never seen -- it only reads turn_id, audio_file,
pause_index, pause_start (and, causally, the already-elapsed durations of
EARLIER pauses in the same turn) to build features. It never reads the
`label` or the current pause's `pause_end` column.
"""
import argparse
import csv
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eot.features import featurize_labels_df  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "eot", "model.joblib"))
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]

    with open(os.path.join(args.data_dir, "labels.csv")) as f:
        rows = list(csv.DictReader(f))

    # predict.py must not depend on the label column even if present
    for r in rows:
        r.pop("label", None)

    X, keys, _ = featurize_labels_df(rows, args.data_dir)
    p = model.predict_proba(X)[:, 1]

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pi_p in zip(keys, p):
            w.writerow([tid, pi, f"{pi_p:.4f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
