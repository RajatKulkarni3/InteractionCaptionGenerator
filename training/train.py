"""
TRAIN + EVALUATE — proper 60/40 train/test split with real, reported
metrics. This is what turns "seems to work" into "measured to work."

WHAT THIS DOES:
  1. Loads features.csv (produced by feature_extraction.py).
  2. Splits 60% train / 40% test, STRATIFIED by class so rare classes
     aren't accidentally starved from one side of the split.
  3. Trains a RandomForestClassifier on the 60% (no GPU needed -- this
     runs in seconds on the engineered features, not raw images).
  4. Evaluates ONLY on the held-out 40%, which the model never saw
     during training -- this is what makes the resulting numbers a
     real reliability measurement instead of a training-accuracy vanity
     metric.
  5. Prints + saves: overall accuracy, a per-class precision/recall/F1
     report (so you can see which specific actions are weak, not just
     one blended number), and a confusion matrix (so you can see what
     it's confusing with what).
  6. Saves the trained model to model.joblib and the evaluation report
     to evaluation_report.txt.

HONEST LIMITS, STATED PLAINLY:
  - This validates the INSTANTANEOUS geometry features (distances,
    angles) extracted from single images. It does NOT validate the
    duration-based rules in interaction_engine.py ("near mouth for
    1.8 seconds") -- there's no time axis in a single-image dataset.
  - A held-out 40% from the SAME dataset only tells you how well this
    generalizes to images similar to that dataset's photography style
    (typically posed, well-lit, uncluttered stock-photo-like images).
    It does not by itself prove reliability on your own webcam/room --
    that needs your own recorded validation clips too, ideally.

USAGE:
    python train.py --features features.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

FAR_SENTINEL = 999.0

# Project root is one level above training/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Train + evaluate the action classifier")
    parser.add_argument(
        "--features", type=str, nargs="+",
        default=[str(_PROJECT_ROOT / "outputs" / "cad_features.csv")],
        help="One or more feature CSVs to train on (they are concatenated)",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.4,
        help="Fraction held out for testing -- 0.4 gives the 60/40 split",
    )
    parser.add_argument(
        "--model_out", type=str,
        default=str(_PROJECT_ROOT / "models" / "cad_model.joblib"),
        help="Where to save the trained model (default: models/cad_model.joblib)",
    )
    parser.add_argument(
        "--report_out", type=str,
        default=str(_PROJECT_ROOT / "outputs" / "evaluation_report.txt"),
        help="Where to save the evaluation report",
    )
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    # Load and concatenate all feature CSVs
    dfs = []
    for csv_path in args.features:
        if os.path.exists(csv_path):
            part = pd.read_csv(csv_path)
            print(f"Loaded {len(part)} rows from {csv_path}")
            dfs.append(part)
        else:
            print(f"Warning: {csv_path} not found, skipping.")
    
    if not dfs:
        print("Error: No valid feature CSVs found. Exiting.")
        sys.exit(1)
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataset: {len(df)} total rows from {len(dfs)} file(s).")
    
    AFFORDANCE_COLS = ["openable", "cuttable", "pourable", "containable", "supportable", "holdable"]
    is_cad = all(c in df.columns for c in AFFORDANCE_COLS)
    
    if is_cad:
        feature_cols = [c for c in df.columns if c not in AFFORDANCE_COLS]
        X = df[feature_cols]  # Keep as DataFrame for ColumnTransformer
        y = df[AFFORDANCE_COLS].values
        labels_sorted = AFFORDANCE_COLS
        strat = None
        print(f"Loaded {len(df)} multi-label examples for CAD affordances.")
    else:
        feature_cols = [c for c in df.columns if c not in ("image_path", "label")]
        X = df[feature_cols]  # Keep as DataFrame
        y = df["label"].values
        labels_sorted = sorted(df["label"].unique())
        strat = y
        print(f"Loaded {len(df)} labeled examples across {df['label'].nunique()} classes:")
        print(df["label"].value_counts().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=strat,
        random_state=args.random_state,
    )
    print(f"\nSplit: {len(X_train)} train ({100 * (1 - args.test_size):.0f}%), "
          f"{len(X_test)} test ({100 * args.test_size:.0f}%)")

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.model_selection import GridSearchCV

    # Automatically identify categorical vs numeric columns
    categorical_cols = ["object_class"]
    numeric_cols = [c for c in feature_cols if c != "object_class"]

    # Build the preprocessor: One-hot encode the category, scale the geometric distances
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value=FAR_SENTINEL)),
                ("scaler", StandardScaler())
            ]), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )

    from sklearn.multioutput import ClassifierChain

    base_rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=args.random_state,
    )
    
    # Use ClassifierChain to explicitly capture label correlations 
    # instead of independent models, to boost exact match accuracy.
    clf = ClassifierChain(base_rf) if is_cad else base_rf

    # The full pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf),
    ])

    # GridSearchCV to find the best hyperparameters, optimized for imbalanced classes
    param_grid = {
        "clf__estimator__n_estimators": [100, 300],
        "clf__estimator__max_depth": [None, 15, 20]
    } if is_cad else {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [None, 15, 20]
    }

    # scoring="f1_macro" ensures we care equally about rare classes (like cuttable)
    search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=3, 
        scoring="f1_macro" if is_cad else "accuracy",
        n_jobs=-1
    )

    print("Running GridSearchCV to optimize hyperparameters...")
    search.fit(X_train, y_train)
    pipeline = search.best_estimator_
    print(f"Best hyperparameters found: {search.best_params_}")

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if is_cad:
        report = classification_report(y_test, y_pred, target_names=labels_sorted, zero_division=0)
        cm_str = "Confusion matrix not supported for MultiOutputClassifier."
        # ClassifierChain estimators have different input shapes. Just take the first one's importances for original features.
        importances = pipeline.named_steps["clf"].estimators_[0].feature_importances_
    else:
        report = classification_report(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
        cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
        cm_str = cm_df.to_string()
        importances = pipeline.named_steps["clf"].feature_importances_

    print(f"\n=== TEST-SET ACCURACY (on the 40% the model never trained on): {accuracy:.1%} ===\n")
    print(report)
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(cm_str)

    importance_lines = sorted(
        zip(feature_cols, importances), key=lambda t: t[1], reverse=True
    )
    print("\nFeature importances (averaged if MultiOutput):")
    for name, imp in importance_lines:
        print(f"  {name}: {imp:.3f}")

    with open(args.report_out, "w") as f:
        f.write(f"Train/test split: {100 * (1 - args.test_size):.0f}/{100 * args.test_size:.0f} "
                 f"({len(X_train)} train, {len(X_test)} test)\n\n")
        f.write(f"Test-set accuracy: {accuracy:.1%}\n\n")
        f.write("Per-class report:\n")
        f.write(report)
        f.write("\nConfusion matrix (rows = actual, columns = predicted):\n")
        f.write(cm_str)
        f.write("\n\nFeature importances (averaged if MultiOutput):\n")
        for name, imp in importance_lines:
            f.write(f"  {name}: {imp:.3f}\n")

    joblib.dump({"pipeline": pipeline, "feature_columns": feature_cols, "labels": labels_sorted}, args.model_out)
    print(f"\nSaved trained model to {args.model_out}")
    print(f"Saved full evaluation report to {args.report_out}")


if __name__ == "__main__":
    main()
