"""
VIDEO FEATURE EXTRACTION -- turns a labeled VIDEO dataset into a CSV of
engineered geometric + DURATION features for training a real, evaluated
action classifier that can validate the timing logic in
interaction_engine.py (e.g. "near mouth for 1.5 seconds").

EXPECTED INPUT LAYOUT (matches HMDB51 after extraction):
    dataset_root/
        drink/
            clip001.avi
            clip002.avi
            ...
        eat/
            ...
        sit/
            ...
        ...

For every video, this:
  1. Samples frames at a fixed rate (--sample_fps, default 5/sec -- video
     is usually 25-30fps, so this cuts compute ~5-6x without losing the
     motion that matters for these actions).
  2. Runs your existing YOLO11-Pose model + objectDetection.py on each
     sampled frame (same functions feature_extraction.py uses on images).
  3. Computes the SAME instantaneous geometric features as
     feature_extraction.py (wrist/face/hip distances, joint angles).
  4. ALSO replicates interaction_engine.py's live proximity+duration
     logic frame-by-frame: using the SAME HAND_NEAR_OBJECT_FRAC (0.35)
     and MOUTH_NEAR_OBJECT_FRAC (0.28) thresholds, it tracks the longest
     unbroken run of "object near wrist/mouth" per category, in seconds.
     This is the piece single images structurally cannot produce -- it's
     the actual reason to use video.
  5. Collapses each video into ONE row: video_path, label, then
     mean/min of each instantaneous feature across sampled frames, plus
     max sustained-duration (seconds) per object category.

Column name note: the video path is written under the "image_path"
column on purpose, so train.py (which excludes "image_path" and
"label" and treats everything else as a feature) works UNCHANGED on
this file too.

USAGE:
    python video_feature_extraction.py --dataset_root /path/to/dataset_root \
        --out video_features.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2  # type: ignore (if IDE is caching)
import scipy.io  # type: ignore (if IDE is caching)

# Add the project root to sys.path so src.pose_estimation and
# src.object_detection are importable from the training/ directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pose_estimation import (  # noqa: E402
    _POSE_MODEL,
    pose_result_to_person,
)
from src.object_detection import detect_objects, set_classes, _DEFAULT_CLASSES  # noqa: E402

# Reuse the exact per-frame feature function + column list + object
# category map from the image-based script, so train/eval features
# are computed identically whether they came from an image or a video
# frame.
from feature_extraction import (  # noqa: E402
    extract_features,
    FEATURE_COLUMNS,
    RELEVANT_OBJECT_CLASSES,
    FAR_SENTINEL,
)

# Same thresholds interaction_engine.py uses live -- duplicated here
# (not imported) because interaction_engine.py's version is wired to
# its live object-tracker class, not a simple per-frame distance check.
# If you change these in interaction_engine.py, change them here too.
HAND_NEAR_OBJECT_FRAC = 0.35
MOUTH_NEAR_OBJECT_FRAC = 0.28

OBJECT_CATEGORIES = sorted(set(RELEVANT_OBJECT_CLASSES.values()))

VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv", ".webm")


class DurationTracker:
    """Tracks, per object category, the longest unbroken run of
    'wrist near object' and 'face near object' across a video's
    sampled frames, converting frame-run-length to seconds using the
    effective sampled frame rate."""

    def __init__(self, categories, effective_fps: float):
        self.effective_fps = max(effective_fps, 1e-6)
        self.wrist_streak = {c: 0 for c in categories}
        self.wrist_best = {c: 0 for c in categories}
        self.face_streak = {c: 0 for c in categories}
        self.face_best = {c: 0 for c in categories}

    def update(self, feats: dict):
        for category in self.wrist_streak:
            wrist_key = f"wrist_dist_{category}"
            wrist_val = feats.get(wrist_key, FAR_SENTINEL)
            if wrist_val < HAND_NEAR_OBJECT_FRAC:
                self.wrist_streak[category] += 1
                self.wrist_best[category] = max(self.wrist_best[category], self.wrist_streak[category])
            else:
                self.wrist_streak[category] = 0

            face_key = f"face_dist_{category}"
            face_val = feats.get(face_key, FAR_SENTINEL)
            if face_val < MOUTH_NEAR_OBJECT_FRAC:
                self.face_streak[category] += 1
                self.face_best[category] = max(self.face_best[category], self.face_streak[category])
            else:
                self.face_streak[category] = 0

    def durations(self) -> dict:
        out = {}
        for category in self.wrist_streak:
            out[f"max_wrist_near_{category}_sec"] = self.wrist_best[category] / self.effective_fps
            out[f"max_face_near_{category}_sec"] = self.face_best[category] / self.effective_fps
        return out


DURATION_COLUMNS = []
for _cat in OBJECT_CATEGORIES:
    DURATION_COLUMNS.append(f"max_wrist_near_{_cat}_sec")
    DURATION_COLUMNS.append(f"max_face_near_{_cat}_sec")

AGG_COLUMNS = [f"mean_{c}" for c in FEATURE_COLUMNS] + [f"min_{c}" for c in FEATURE_COLUMNS] + DURATION_COLUMNS


def process_video(video_path: Path, pose_model, imgsz: int, sample_fps: float):
    """
    Process one HMDB51 video represented as a folder of JPEG frames.
    """

    frame_paths = sorted(
        video_path.glob("*.jpg"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
    )

    if not frame_paths:
        return None

    # HMDB51 frame-folder datasets do not contain FPS metadata.
    # Use 30 FPS as the source rate.
    src_fps = 30.0

    frame_interval = max(
        int(round(src_fps / sample_fps)),
        1
    )

    effective_fps = src_fps / frame_interval

    tracker = DurationTracker(
        OBJECT_CATEGORIES,
        effective_fps
    )

    per_frame_feats = []

    for frame_idx, frame_path in enumerate(frame_paths):

        # Sample frames according to --sample_fps
        if frame_idx % frame_interval != 0:
            continue

        frame = cv2.imread(str(frame_path))

        if frame is None:
            continue

        pose_results = pose_model(
            frame,
            verbose=False,
            imgsz=imgsz
        )

        person = pose_result_to_person(
            pose_results[0]
        )

        if person is None:
            continue

        detections = detect_objects(
            frame,
            imgsz=imgsz
        )["detections"]

        feats = extract_features(
            person,
            detections
        )

        per_frame_feats.append(feats)

        tracker.update(feats)

    if not per_frame_feats:
        return None

    row = {}

    for col in FEATURE_COLUMNS:
        values = [
            f[col]
            for f in per_frame_feats
        ]

        row[f"mean_{col}"] = sum(values) / len(values)
        row[f"min_{col}"] = min(values)

    row.update(
        tracker.durations()
    )

    return row


def main():
    parser = argparse.ArgumentParser(description="Extract duration-aware features from a labeled video dataset")
    parser.add_argument("--dataset_root", type=str, required=True,
                         help="Folder containing Penn Action Dataset (with /frames and /labels)")
    parser.add_argument("--out", type=str,
                         default=str(Path(__file__).resolve().parent.parent / "outputs" / "penn_video_features.csv"),
                         help="Output CSV path (default: outputs/penn_video_features.csv in project root)")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--sample_fps", type=float, default=5.0,
                         help="How many frames per second of video to actually process")
    parser.add_argument("--limit_per_class", type=int, default=None,
                         help="Optional cap per class, useful for a fast first pass")
    args = parser.parse_args()

    if detect_objects is None:
        raise RuntimeError(
            "src.object_detection could not be imported. "
            "Run from the project root or ensure src/ is on PYTHONPATH."
        )

    pose_model = _POSE_MODEL  # reuse the module singleton — no second model load
    print("Pose model ready (loaded via src.pose_estimation singleton).")

    # Setup YOLO-World classes for Penn Action
    penn_classes = ["baseball", "baseball bat", "barbell", "bench", "bowling ball", "guitar", "jump rope", "tennis racket"]
    set_classes(_DEFAULT_CLASSES + penn_classes)
    print("YOLO-World ready with Penn Action vocabulary.")

    dataset_root = Path(args.dataset_root)
    labels_dir = dataset_root / "labels"
    frames_dir = dataset_root / "frames"
    
    if not labels_dir.exists() or not frames_dir.exists():
        raise RuntimeError(f"Expected 'labels' and 'frames' subdirectories in {dataset_root}")

    mat_files = sorted(labels_dir.glob("*.mat"))
    if not mat_files:
        raise RuntimeError(f"No .mat files found in {labels_dir}")
        
    if args.limit_per_class:
        mat_files = mat_files[: args.limit_per_class]

    rows_written = 0
    skipped_no_detection = 0

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"] + AGG_COLUMNS)

        for mat_path in mat_files:
            video_id = mat_path.stem
            video_path = frames_dir / video_id
            
            if not video_path.is_dir() or not any(video_path.glob("*.jpg")):
                print(f"Skipping {video_id}: No frames found.")
                continue
                
            try:
                mat = scipy.io.loadmat(str(mat_path))
                label = str(mat['action'][0])
            except Exception as e:
                print(f"Skipping {video_id}: Error reading label ({e})")
                continue

            row = process_video(video_path, pose_model, args.imgsz, args.sample_fps)
            if row is None:
                skipped_no_detection += 1
                continue
            writer.writerow([str(video_path), label] + [row[c] for c in AGG_COLUMNS])
            rows_written += 1
            print(f"Processed {video_id} -> {label}")

    print(f"\nWrote {rows_written} rows to {args.out}")
    print(f"Skipped {skipped_no_detection} videos with no usable person detection in any sampled frame")


if __name__ == "__main__":
    main()
