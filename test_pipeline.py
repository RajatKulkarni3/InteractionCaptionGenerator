"""
Static image pipeline smoke test.
Run from the project root:
    python test_pipeline.py
"""
import sys
import cv2
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "training")

from pose_estimation import estimate_pose
from object_detection import detect_objects, split_persons_and_objects
from feature_extraction import extract_features, FEATURE_COLUMNS, FAR_SENTINEL
import joblib

# ------------------------------------------------------------------
# Load test image — replace this path with any real photo if you want
# to see actual detections. Falls back to a blank frame.
# ------------------------------------------------------------------
IMG_PATH = r"C:\Users\alank\OneDrive\Pictures\Camera Roll 1\WIN_20260905_16_41_14_Pro.jpg"

frame = cv2.imread(IMG_PATH)
if frame is None:
    print(f"[INFO] Could not read {IMG_PATH!r} — using a blank 480x640 frame instead.")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
else:
    print(f"[INFO] Loaded image: {frame.shape[1]}x{frame.shape[0]} px")

# ------------------------------------------------------------------
# 1. Pose estimation
# ------------------------------------------------------------------
print("\n--- Pose estimation ---")
pose_result = estimate_pose(frame)
person      = pose_result["person_pose"]
named_kps   = pose_result["keypoints"]
print("Person detected :", person is not None)
if named_kps:
    print("Named keypoints  :", list(named_kps.keys()))

# ------------------------------------------------------------------
# 2. Object detection
# ------------------------------------------------------------------
print("\n--- Object detection ---")
det_result  = detect_objects(frame, conf_threshold=0.20)
detections  = det_result["detections"]
_, objects  = split_persons_and_objects(detections)
print("All detections   :", [d["object"] for d in detections])
print("Non-person objs  :", [o["object"] for o in objects])

# ------------------------------------------------------------------
# 3. Feature extraction
# ------------------------------------------------------------------
print("\n--- Feature extraction ---")
if person is not None:
    geo = extract_features(person, objects)
    print("Feature columns  :", FEATURE_COLUMNS)
    print("Values           :", [round(geo[c], 3) for c in FEATURE_COLUMNS])
    near = {c: v for c, v in geo.items() if v != FAR_SENTINEL}
    print("Non-sentinel feats:", near)
else:
    print("Skipped — no person in frame")

# ------------------------------------------------------------------
# 4. RF model classification
# ------------------------------------------------------------------
print("\n--- RF classification ---")
bundle = joblib.load("models/model.joblib")
print("Labels           :", bundle["labels"])
print("Feature cols     :", len(bundle["feature_columns"]))

if person is not None:
    geo = extract_features(person, objects)
    feat_cols = bundle["feature_columns"]
    row = []
    for col in feat_cols:
        if col.startswith("mean_") or col.startswith("min_"):
            base = col[5:]
            row.append(geo.get(base, FAR_SENTINEL))
        elif col.startswith("max_wrist_near_") and col.endswith("_sec"):
            row.append(0.0)   # no duration info from a single image
        elif col.startswith("max_face_near_") and col.endswith("_sec"):
            row.append(0.0)
        else:
            row.append(geo.get(col, FAR_SENTINEL))  # raw feature name

    X     = np.array([row], dtype=np.float64)
    pipe  = bundle["pipeline"]
    proba = pipe.predict_proba(X)[0]
    idx   = int(proba.argmax())
    label = bundle["labels"][idx]
    conf  = round(float(proba[idx]) * 100, 1)
    print(f"Prediction       : {label}  ({conf}%)")
    top3  = sorted(zip(bundle["labels"], proba), key=lambda t: t[1], reverse=True)[:3]
    print("Top-3            :", [(l, round(p*100,1)) for l, p in top3])
else:
    print("Skipped — no person in frame")

print("\n=== Smoke test complete ===")
