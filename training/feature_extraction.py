"""
FEATURE EXTRACTION — turns a labeled image dataset into a CSV of
engineered geometric features for training a real, evaluated action
classifier.

EXPECTED INPUT LAYOUT (matches the HAR Kaggle/Hugging Face dataset):
    dataset_root/
        drinking/
            img001.jpg
            img002.jpg
            ...
        calling/
            ...
        sitting/
            ...
        ...

For every image, this:
  1. Runs your existing YOLO11-Pose model to get the person's keypoints
     (pose_result_to_person from pose_estimation_final.py).
  2. Runs your existing object detector (detect_objects from
     objectDetection.py) to get object boxes.
  3. Computes the SAME kind of geometric features interaction_engine.py
     uses live -- wrist-to-object distance, face-to-object distance,
     knee/thigh/torso angles -- normalized by bbox_height so they don't
     need re-tuning per image resolution or camera distance.
  4. Writes one row per successfully-detected person to features.csv:
     image_path, label, then the feature columns.

IMPORTANT, STATED PLAINLY:
This dataset is single IMAGES, not video -- there is no time axis, so
none of the DURATION features (like "near mouth for 1.8 seconds") that
interaction_engine.py uses live can be trained or validated here. This
script only extracts the instantaneous-geometry features. Training on
this gets you a real, measurable single-frame classifier; the duration
logic stays rule-based (as already built) until you have a video
dataset (e.g. NTU RGB+D) to validate it against.

USAGE:
    python feature_extraction.py --dataset_root /path/to/dataset_root \
        --out features.csv
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2

# --- Reuse the SAME geometry code the live pipeline uses, so training
# and inference are computing identical features. ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pose_estimation_final import (  # noqa: E402
    pose_result_to_person,
    get_interaction_keypoints,
    get_face_point,
    knee_angle,
    thigh_angle_from_vertical,
    torso_angle_from_vertical,
)

try:
    from objectDetection import detect_objects  # noqa: E402
except ImportError:
    detect_objects = None  # allows --dry_run smoke testing without your objectDetection.py present


# Object classes relevant to each target action -- extend this as you
# add more actions. Matches the COCO class names your detector already
# outputs.
RELEVANT_OBJECT_CLASSES = {
    "cup": "drink_related",
    "wine glass": "drink_related",
    "bottle": "drink_related",
    "cell phone": "phone_related",
    "laptop": "laptop_related",
    "chair": "chair_related",
    "book": "reading_related",
}

FAR_SENTINEL = 999.0  # "no such object detected nearby" -- large, but finite, so a tree/forest
                        # classifier can still split on it meaningfully instead of choking on NaN

FEATURE_COLUMNS = [
    "left_knee_angle", "right_knee_angle",
    "left_thigh_angle", "right_thigh_angle",
    "torso_tilt", "head_above_shoulders",
    "wrist_dist_drink_related", "face_dist_drink_related",
    "wrist_dist_phone_related", "face_dist_phone_related",
    "wrist_dist_laptop_related", "face_dist_laptop_related",
    "hip_dist_chair_related",
]


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def extract_features(person, detections) -> dict:
    """person: PersonPose from pose_result_to_person.
    detections: list of {"object": cls, "confidence": float, "bbox": [x1,y1,x2,y2]}
    Returns a dict matching FEATURE_COLUMNS (FAR_SENTINEL where evidence is absent)."""
    bbox_height = person.bbox_height

    feats = {col: FAR_SENTINEL for col in FEATURE_COLUMNS}

    lk = knee_angle(person, "left")
    rk = knee_angle(person, "right")
    lt = thigh_angle_from_vertical(person, "left")
    rt = thigh_angle_from_vertical(person, "right")
    tt = torso_angle_from_vertical(person)
    feats["left_knee_angle"] = lk if lk is not None else FAR_SENTINEL
    feats["right_knee_angle"] = rk if rk is not None else FAR_SENTINEL
    feats["left_thigh_angle"] = lt if lt is not None else FAR_SENTINEL
    feats["right_thigh_angle"] = rt if rt is not None else FAR_SENTINEL
    feats["torso_tilt"] = tt if tt is not None else FAR_SENTINEL

    interaction_kps = get_interaction_keypoints(person)
    wrist_points = [(x, y) for x, y, kind in interaction_kps if kind == "wrist"]
    hip_points = [(x, y) for x, y, kind in interaction_kps if kind == "hip"]
    face_point = get_face_point(person)
    from pose_estimation_final import _head_above_shoulders
    hu = _head_above_shoulders(person)
    feats["head_above_shoulders"] = {True: 1.0, False: 0.0, None: -1.0}[hu]

    # group detected objects by relevance category, then take the
    # closest one per category as the feature (closest = most likely
    # to be the one actually being interacted with)
    by_category = {}
    for det in detections:
        category = RELEVANT_OBJECT_CLASSES.get(det["object"])
        if category is None:
            continue
        x1, y1, x2, y2 = det["bbox"]
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        by_category.setdefault(category, []).append(center)

    for category, centers in by_category.items():
        if wrist_points:
            min_wrist_dist = min(_dist(c, w) for c in centers for w in wrist_points) / bbox_height
            key = f"wrist_dist_{category}"
            if key in feats:
                feats[key] = min_wrist_dist
        if face_point is not None:
            min_face_dist = min(_dist(c, face_point) for c in centers) / bbox_height
            key = f"face_dist_{category}"
            if key in feats:
                feats[key] = min_face_dist
        if hip_points:
            min_hip_dist = min(_dist(c, h) for c in centers for h in hip_points) / bbox_height
            key = f"hip_dist_{category}"
            if key in feats:
                feats[key] = min_hip_dist

    return feats


def main():
    parser = argparse.ArgumentParser(description="Extract geometric features from a labeled image dataset")
    parser.add_argument("--dataset_root", type=str, required=True,
                         help="Folder containing one subfolder per class, each full of images")
    parser.add_argument("--out", type=str, default="features.csv")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--limit_per_class", type=int, default=None,
                         help="Optional cap per class, useful for a fast first pass")
    args = parser.parse_args()

    if detect_objects is None:
        raise RuntimeError(
            "objectDetection.py (with a detect_objects function) was not importable. "
            "Run this from your project folder (next to objectDetection.py), or add it "
            "to PYTHONPATH."
        )

    from ultralytics import YOLO
    print("Loading YOLO11-Pose model...")
    pose_model = YOLO("yolo11n-pose.pt")

    dataset_root = Path(args.dataset_root)
    class_dirs = sorted(p for p in dataset_root.iterdir() if p.is_dir())
    if not class_dirs:
        raise RuntimeError(f"No class subfolders found under {dataset_root}")

    rows_written = 0
    skipped_no_person = 0

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"] + FEATURE_COLUMNS)

        for class_dir in class_dirs:
            label = class_dir.name
            image_paths = sorted(
                p for p in class_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            if args.limit_per_class:
                image_paths = image_paths[: args.limit_per_class]

            print(f"[{label}] {len(image_paths)} images")
            for img_path in image_paths:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue

                pose_results = pose_model(frame, verbose=False, imgsz=args.imgsz)
                person = pose_result_to_person(pose_results[0])
                if person is None:
                    skipped_no_person += 1
                    continue

                detections = detect_objects(frame, imgsz=args.imgsz)
                feats = extract_features(person, detections)

                writer.writerow([str(img_path), label] + [feats[c] for c in FEATURE_COLUMNS])
                rows_written += 1

    print(f"\nWrote {rows_written} rows to {args.out}")
    print(f"Skipped {skipped_no_person} images with no detected person")


if __name__ == "__main__":
    main()
