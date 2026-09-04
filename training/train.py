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

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

FAR_SENTINEL = 999.0


def main():
    parser = argparse.ArgumentParser(description="Train + evaluate the action classifier")
    parser.add_argument("--features", type=str, default="features.csv")
    parser.add_argument("--test_size", type=float, default=0.4,
                         help="Fraction held out for testing -- 0.4 gives the 60/40 split you asked for")
    parser.add_argument("--model_out", type=str, default="model.joblib")
    parser.add_argument("--report_out", type=str, default="evaluation_report.txt")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    feature_cols = [c for c in df.columns if c not in ("image_path", "label")]

    X = df[feature_cols].values
    y = df["label"].values

    print(f"Loaded {len(df)} labeled examples across {df['label'].nunique()} classes:")
    print(df["label"].value_counts().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )
    print(f"\nSplit: {len(X_train)} train ({100 * (1 - args.test_size):.0f}%), "
          f"{len(X_test)} test ({100 * args.test_size:.0f}%)")

    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=FAR_SENTINEL)),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",  # don't let a class with more images dominate
            random_state=args.random_state,
        )),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(df["label"].unique()))
    labels_sorted = sorted(df["label"].unique())

    print(f"\n=== TEST-SET ACCURACY (on the 40% the model never trained on): {accuracy:.1%} ===\n")
    print(report)

    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(cm_df.to_string())

    # feature importances -- tells you which signals are actually
    # doing the work, useful for deciding what to improve next
    importances = pipeline.named_steps["clf"].feature_importances_
    importance_lines = sorted(
        zip(feature_cols, importances), key=lambda t: t[1], reverse=True
    )
    print("\nFeature importances:")
    for name, imp in importance_lines:
        print(f"  {name}: {imp:.3f}")

    with open(args.report_out, "w") as f:
        f.write(f"Train/test split: {100 * (1 - args.test_size):.0f}/{100 * args.test_size:.0f} "
                 f"({len(X_train)} train, {len(X_test)} test)\n\n")
        f.write(f"Test-set accuracy: {accuracy:.1%}\n\n")
        f.write("Per-class report:\n")
        f.write(report)
        f.write("\nConfusion matrix (rows = actual, columns = predicted):\n")
        f.write(cm_df.to_string())
        f.write("\n\nFeature importances:\n")
        for name, imp in importance_lines:
            f.write(f"  {name}: {imp:.3f}\n")

    joblib.dump({"pipeline": pipeline, "feature_columns": feature_cols, "labels": labels_sorted}, args.model_out)
    print(f"\nSaved trained model to {args.model_out}")
    print(f"Saved full evaluation report to {args.report_out}")


if __name__ == "__main__":
    main()
