"""Train the EOT classifier and save it for predict.py.

    python -m eot.train --data_dirs eot_data/english eot_data/hindi \
        --model_out eot/model.joblib

Trains on BOTH language folders pooled together (features are prosodic /
language-agnostic on purpose), evaluates with grouped cross-validation
(never splitting a turn across folds), then refits on all data and saves
the pipeline (StandardScaler + classifier) to disk.
"""
import argparse
import csv
import os

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from eot.features import FEATURE_NAMES, featurize_labels_df


def load_rows(data_dir):
    with open(os.path.join(data_dir, "labels.csv")) as f:
        return list(csv.DictReader(f))


def auc_score(y, s):
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", required=True)
    ap.add_argument("--model_out", default="eot/model.joblib")
    args = ap.parse_args()

    all_X, all_y, all_groups, all_keys = [], [], [], []
    for d in args.data_dirs:
        rows = load_rows(d)
        X, keys, y = featurize_labels_df(rows, d)
        lang_tag = os.path.basename(os.path.normpath(d))
        groups = [f"{lang_tag}:{tid}" for tid, _ in keys]
        all_X.append(X)
        all_y.append(y)
        all_groups.extend(groups)
        all_keys.extend(keys)
        print(f"{d}: {X.shape[0]} pauses from {len(set(groups))} turns")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.array(all_groups)

    candidates = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0)),
        "gboost": GradientBoostingClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=0),
        "svc_rbf": make_pipeline(StandardScaler(), SVC(
            probability=True, class_weight="balanced", C=1.0, gamma="scale")),
    }

    gkf = GroupKFold(n_splits=5)
    print("\n--- 5-fold grouped cross-validation (by turn) ---")
    best_name, best_auc = None, -1
    for name, model in candidates.items():
        probs = cross_val_predict(model, X, y, cv=gkf, groups=groups,
                                   method="predict_proba")[:, 1]
        auc = auc_score(y, probs)
        acc = np.mean((probs >= 0.5).astype(int) == y)
        print(f"{name:10s}  AUC={auc:.3f}  acc@0.5={acc:.3f}")
        if auc > best_auc:
            best_auc, best_name = auc, name

    print(f"\nSelected model: {best_name} (cv AUC={best_auc:.3f})")
    final_model = candidates[best_name]
    final_model.fit(X, y)

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump({"model": final_model, "feature_names": FEATURE_NAMES,
                 "model_name": best_name}, args.model_out)
    print(f"saved -> {args.model_out}")


if __name__ == "__main__":
    main()
