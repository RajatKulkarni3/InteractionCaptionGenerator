"""
PROJECT V2C — Unified live pipeline
=====================================
Wires together:
  • YOLO-World object detector  (src/object_detection.py)
  • YOLO11-Pose estimator        (src/pose_estimation.py)
  • Interaction engine            (src/interaction_engine.py)
  • RandomForest action model    (models/model.joblib)

On-screen output per frame
--------------------------
  Posture: <label>  (<stability>% stable)         ← red
  Action:  <label>                   [<conf>%]    ← green, bold
  • <reason line 1>                               ← green
  • <reason line 2>                               ← green
  • <reason line 3>                               ← green

Rule-engine priority
--------------------
  1. Interaction engine fired → its action / reasons / confidence
     (rule engine always wins: it has TEMPORAL evidence the RF cannot)
  2. RF model prediction + proximity reasons
     (used while temporal evidence is still building)
  3. No action (neither fired, or no person in frame)

Feature vector for RF
---------------------
  Built from the same ``extract_features()`` used during training.
  Single-frame approximation of the aggregated video features:
    mean_X / min_X  → current frame's instantaneous value of X
    max_wrist_near_*_sec  → proximity_log duration for wrist keypoints
    max_face_near_*_sec   → proximity_log duration for face keypoints
  Filled with FAR_SENTINEL (999) when not determinable this frame.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — make sure training/ and project root are importable
# ---------------------------------------------------------------------------
_HERE         = Path(__file__).resolve().parent        # …/src
_PROJECT_ROOT = _HERE.parent                           # …/InteractionCaptionGenerator
_TRAINING_DIR = _PROJECT_ROOT / "training"

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

# ---------------------------------------------------------------------------
# Pipeline imports (models load as module-level singletons)
# ---------------------------------------------------------------------------
from object_detection import detect_objects, split_persons_and_objects, set_classes, _DEFAULT_CLASSES  # noqa: E402
from interaction_engine import InteractionEngine                          # noqa: E402
from pose_estimation import (                                             # noqa: E402
    estimate_pose,
    pose_result_to_person,
    get_interaction_keypoints,
    get_face_point,
    PostureClassifier,
    knee_angle,
    thigh_angle_from_vertical,
    torso_angle_from_vertical,
)

# feature_extraction lives in training/ (already on sys.path above)
from feature_extraction import (                                          # noqa: E402
    extract_features,
    FEATURE_COLUMNS,
    RELEVANT_OBJECT_CLASSES,
    FAR_SENTINEL,
)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Master YOLO-World vocabulary — combines COCO, CAD-120, and Penn Action
_MASTER_CLASSES = [
    "person", "cup", "bottle", "wine glass", "cell phone", "laptop", "book",
    "chair", "table", "kettle", "plate", "knife", "medicine box", "can",
    "microwave", "bowl", "mug", "baseball bat", "guitar", "tennis racket",
    "barbell",
]
set_classes(_MASTER_CLASSES)
print(f"YOLO-World vocabulary set to {len(_MASTER_CLASSES)} master classes.")

parser = argparse.ArgumentParser(description="Human Action Recognition Pipeline")
parser.add_argument("--input", default="0", help="Camera index or path to video file (default: 0)")
parser.add_argument("--output", default="", help="Path to save annotated output video (e.g., output.mp4)")
args = parser.parse_args()

try:
    SOURCE = int(args.input)
except ValueError:
    SOURCE = args.input

OUTPUT_FILE = args.output

INFERENCE_IMGSZ               = 640
OBJECT_DETECTION_EVERY_N_FRAMES = 2
DEBUG_EVERY_N_FRAMES          = 30

# Rolling feature buffer for the RF classifier
# -------------------------------------------------
# The RF was trained on video-aggregated features (mean_X, min_X per
# FEATURE_COLUMN across a ~5-second clip).  Passing a single frame's
# instantaneous values produces numerical distributions that are alien
# to the classifier and cause wild mispredictions.
#
# Fix: accumulate the last RF_BUFFER_MAXLEN per-frame geometry dicts and
# compute the true mean / min before every RF call.  At 30 fps this is
# ~5 seconds of history — matching the training distribution.
RF_BUFFER_MAXLEN  = 150                       # ~5 s at 30 fps
_track_feature_buffers: dict[int, deque] = {} # Map track_id -> deque of (wrist_dist, face_dist)


# ---------------------------------------------------------------------------
# Load the retrained RandomForest model
# ---------------------------------------------------------------------------

_MODEL_PATH = _PROJECT_ROOT / "models" / "cad_model.joblib"
_rf_bundle: Optional[dict] = None

if _MODEL_PATH.exists():
    try:
        _rf_bundle = joblib.load(str(_MODEL_PATH))
        _feature_cols: list[str] = _rf_bundle["feature_columns"]
        _labels: list[str]       = _rf_bundle["labels"]
        print(f"RF model loaded from {_MODEL_PATH}  "
              f"({len(_labels)} classes, {len(_feature_cols)} features)")
    except Exception as exc:
        print(f"[WARNING] Could not load RF model from {_MODEL_PATH}: {exc}")
        _rf_bundle = None
else:
    print(f"[WARNING] RF model not found at {_MODEL_PATH}. "
          "Run  python training/train.py  to create it. "
          "Falling back to rule-engine-only classification.")


# ---------------------------------------------------------------------------
# Stateful objects (one per session)
# ---------------------------------------------------------------------------

posture_classifier = PostureClassifier()
interaction_engine = InteractionEngine()


# ---------------------------------------------------------------------------
# RF feature vector builder
# ---------------------------------------------------------------------------

def _get_cad_features_for_track(
    person,
    track,
    proximity_log: list,
    frame_shape: tuple,
) -> Optional[dict]:
    """Builds the feature dictionary for affordance prediction for a single object track."""
    if _rf_bundle is None or person is None:
        return None

    bbox_height = max(person.bbox_height, 1.0)
    x1, y1, x2, y2 = track.bbox
    xc, yc = track.center()
    
    fh, fw = frame_shape[:2]
    obj_width_norm = (x2 - x1) / fw
    obj_height_norm = (y2 - y1) / fh
    
    # 1. Compute instantaneous distances
    wrist_dist = FAR_SENTINEL
    for side in ("left_wrist", "right_wrist"):
        if side in person.keypoints:
            wx, wy = person.keypoints[side]
            d = math.hypot(wx - xc, wy - yc)
            wrist_dist = min(wrist_dist, d / bbox_height)
            
    face_dist = FAR_SENTINEL
    for kp in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
        if kp in person.keypoints:
            fx, fy = person.keypoints[kp]
            d = math.hypot(fx - xc, fy - yc)
            face_dist = min(face_dist, d / bbox_height)
            
    shoulder_dist = FAR_SENTINEL
    for kp in ("left_shoulder", "right_shoulder"):
        if kp in person.keypoints:
            sx, sy = person.keypoints[kp]
            d = math.hypot(sx - xc, sy - yc)
            shoulder_dist = min(shoulder_dist, d / bbox_height)
            
    hip_dist = FAR_SENTINEL
    for kp in ("left_hip", "right_hip"):
        if kp in person.keypoints:
            hx, hy = person.keypoints[kp]
            d = math.hypot(hx - xc, hy - yc)
            hip_dist = min(hip_dist, d / bbox_height)
            
    knee_dist = FAR_SENTINEL
    for kp in ("left_knee", "right_knee"):
        if kp in person.keypoints:
            kx, ky = person.keypoints[kp]
            d = math.hypot(kx - xc, ky - yc)
            knee_dist = min(knee_dist, d / bbox_height)

    # 2. Push to buffer
    tid = track.track_id
    if tid not in _track_feature_buffers:
        _track_feature_buffers[tid] = deque(maxlen=RF_BUFFER_MAXLEN)
    _track_feature_buffers[tid].append({
        "w": wrist_dist, "f": face_dist,
        "s": shoulder_dist, "h": hip_dist, "k": knee_dist
    })
    
    # 3. Compute mean/min
    buf = _track_feature_buffers[tid]
    
    def get_stats(key):
        vals = [entry[key] for entry in buf if entry[key] != FAR_SENTINEL]
        mean_val = sum(vals)/len(vals) if vals else FAR_SENTINEL
        min_val = min(vals) if vals else FAR_SENTINEL
        return mean_val, min_val
        
    mean_w, min_w = get_stats("w")
    mean_f, min_f = get_stats("f")
    mean_s, min_s = get_stats("s")
    mean_h, min_h = get_stats("h")
    mean_k, min_k = get_stats("k")
    
    # 4. Get duration from proximity_log (from previous frame)
    wrist_sec = 0.0
    face_sec = 0.0
    for rec in proximity_log:
        if rec.track_id == tid:
            if rec.keypoint_name in ("left_wrist", "right_wrist"):
                wrist_sec = max(wrist_sec, rec.duration_sec)
            if rec.keypoint_name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
                face_sec = max(face_sec, rec.duration_sec)
    
    return {
        "object_class": track.cls,
        "mean_wrist_dist": mean_w,
        "min_wrist_dist": min_w,
        "max_wrist_near_sec": wrist_sec,
        "mean_face_dist": mean_f,
        "min_face_dist": min_f,
        "max_face_near_sec": face_sec,
        "mean_shoulder_dist": mean_s,
        "min_shoulder_dist": min_s,
        "mean_hip_dist": mean_h,
        "min_hip_dist": min_h,
        "mean_knee_dist": mean_k,
        "min_knee_dist": min_k,
        "obj_width_norm": obj_width_norm,
        "obj_height_norm": obj_height_norm
    }


def _predict_track_affordances(person, track, proximity_log, frame_shape: tuple) -> list[str]:
    """Returns a list of predicted affordances for this track."""
    feat_dict = _get_cad_features_for_track(person, track, proximity_log, frame_shape)
    if feat_dict is None:
        return []
    try:
        import pandas as pd
        df = pd.DataFrame([feat_dict])
        pipeline = _rf_bundle["pipeline"]
        probas = pipeline.predict_proba(df)
        
        active_affs = []
        if isinstance(probas, list):
            # MultiOutputClassifier: list of (n_samples, 2) arrays
            for i, proba in enumerate(probas):
                if proba[0, 1] > 0.5:
                    active_affs.append(_labels[i])
        else:
            # ClassifierChain: single (n_samples, n_labels) array
            # Each column is P(label=1) for that affordance
            for i in range(probas.shape[1]):
                if probas[0, i] > 0.5:
                    active_affs.append(_labels[i])
        return active_affs
    except Exception as e:
        print(f"Error predicting affordance: {e}")
        return []


# ---------------------------------------------------------------------------
# Affordance-to-verb mapping (for natural language captions)
# ---------------------------------------------------------------------------
_AFFORDANCE_VERBS = {
    "pourable":    "pouring",
    "cuttable":    "cutting",
    "openable":    "opening",
    "containable": "filling",
    "supportable": "leaning on", # default, dynamically replaced based on posture
    "holdable":    "holding",
}


# ---------------------------------------------------------------------------
# Rule-engine explanation builder (natural language)
# ---------------------------------------------------------------------------

def build_explanation(
    rf_label:          Optional[str],
    rf_confidence:     float,
    interaction_events: list,
    proximity_log:     list,
    posture:           str,
    track_affordances: dict = None,
) -> tuple[Optional[str], list[str], float]:
    """
    Synthesize ALL active InteractionEvents + posture into a natural-language
    sentence with explainable bullet-point reasons.

    Returns
    -------
    (sentence, [reason, ...], avg_confidence)

    Example sentences:
      "Person is sitting and drinking."                 (single action)
      "Person is standing and using mobile while drinking water."  (multi)
      "Person is sitting."                              (no actions)
    """
    if track_affordances is None:
        track_affordances = {}

    action_items: list[tuple[str, str]] = []  # (verb, object_name)
    all_reasons: list[str] = []
    confidences: list[float] = []

    # 1. Collect ALL interaction engine events (simultaneous actions)
    for event in interaction_events:
        action_items.append((event.action.lower(), event.object_name.lower()))
        all_reasons.extend(event.reasons[:3])
        confidences.append(event.confidence)

    # 2. If no engine events, try RF affordance fallback
    if not action_items and track_affordances:
        AFFORDANCE_PRIORITY = ["cuttable", "pourable", "openable", "containable", "supportable", "holdable"]
        for tid, affs in track_affordances.items():
            if not affs:
                continue
            # Pick the single most specific/informative affordance for this object
            best_aff = min(affs, key=lambda a: AFFORDANCE_PRIORITY.index(a) if a in AFFORDANCE_PRIORITY else 99)
            
            # Map object name from proximity log
            obj_name = "object"
            for rec in proximity_log:
                if rec.track_id == tid:
                    obj_name = rec.object_class
                    break
            
            if best_aff == "supportable":
                verb = "sitting on" if "sitting" in posture.lower() else "leaning on"
            else:
                verb = _AFFORDANCE_VERBS.get(best_aff, best_aff)
                
            item = (verb, obj_name)
            if item not in action_items:
                action_items.append(item)

    # 3. Also extract proximity-based reasons for context
    if not all_reasons:
        seen_objects: set = set()
        for rec in sorted(proximity_log, key=lambda r: r.distance_px):
            key = rec.object_class
            if key in seen_objects or len(all_reasons) >= 4:
                break
            seen_objects.add(key)
            obj = rec.object_class.capitalize()
            if rec.distance_px == 0.0:
                msg = f"{obj} in contact"
                if rec.duration_sec > 0.2:
                    msg += f" ({rec.duration_sec:.1f}s)"
            elif rec.distance_px < 80:
                msg = f"{obj} nearby ({rec.distance_px:.0f}px)"
            else:
                continue  # skip far objects
            all_reasons.append(msg)

    # 4. Build the natural-language sentence
    posture_lower = posture.lower() if posture != "no person detected" else None

    # Deduplicate actions
    unique_items = []
    for item in action_items:
        if item not in unique_items:
            unique_items.append(item)
    action_items = unique_items

    if posture_lower is None:
        sentence = None
    elif not action_items:
        templates = [
            f"The person is currently {posture_lower}.",
            f"They are {posture_lower}.",
            f"Person is {posture_lower}.",
            f"Currently {posture_lower}."
        ]
        sentence = random.choice(templates)
    elif len(action_items) == 1:
        v, o = action_items[0]
        templates = [
            f"The person is {posture_lower} while {v} the {o}.",
            f"Currently {posture_lower} and {v} the {o}.",
            f"The person is {v} the {o} while {posture_lower}."
        ]
        sentence = random.choice(templates)
    else:
        v1, o1 = action_items[0]
        v2, o2 = action_items[1]
        
        templates = [
            f"The person is {v1} the {o1}. They are currently {v2} the {o2} while {posture_lower}.",
            f"While {posture_lower}, the person is {v1} the {o1} and {v2} the {o2}.",
            f"Currently {posture_lower}. The person is {v1} the {o1}, and also {v2} the {o2}."
        ]
        sentence = random.choice(templates)
        
        # If there are more than 2 actions, append them
        if len(action_items) > 2:
            extra = [f"{v} the {o}" for v, o in action_items[2:]]
            sentence += " Also " + ", ".join(extra) + "."

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    # Deduplicate reasons
    unique_reasons: list[str] = []
    seen: set = set()
    for r in all_reasons:
        if r not in seen:
            unique_reasons.append(r)
            seen.add(r)

    return sentence, unique_reasons[:5], round(avg_conf, 1)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _put(frame, text: str, pos: tuple, scale: float, color: tuple,
         thickness: int = 1) -> int:
    """Draw one line of text and return the y-coordinate of the next line."""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)
    line_height = int(scale * 30 + 6)
    return pos[1] + line_height


def _draw_objects(frame, detections: list[dict]) -> None:
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 130, 0), 2)
        _put(frame, f"{det['object']} {det['confidence']:.2f}",
             (x1, max(y1 - 8, 18)), 0.52, (255, 130, 0), 2)


def _draw_hud(frame, posture: str, stability: float,
              action: Optional[str], reasons: list[str],
              confidence: float) -> None:
    """
    Render the heads-up display:
      Line 1: Natural-language caption sentence (green, bold)
      Lines 2+: Bullet-point explainable reasons (lighter green)
    """
    y = 28

    # Posture line (subtle, secondary)
    y = _put(frame,
             f"Posture: {posture}  ({stability}% stable)",
             (10, y), 0.52, (60, 60, 255), 1)

    y += 6

    if action is None:
        _put(frame, "No action detected", (10, y), 0.58, (160, 160, 160), 1)
        return

    # Main caption sentence (primary, bold)
    y = _put(frame, action, (10, y), 0.68, (30, 220, 30), 2)

    # Confidence badge, right-aligned
    if confidence > 0:
        conf_text = f"[{confidence:.0f}%]"
        (tw, _), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        conf_x = frame.shape[1] - tw - 12
        cv2.putText(frame, conf_text, (conf_x, y - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (30, 220, 30), 1, cv2.LINE_AA)

    y += 2

    # Explainable reason lines (bullet points)
    for reason in reasons:
        y = _put(frame, f"  - {reason}", (10, y), 0.45, (30, 200, 30), 1)


# ---------------------------------------------------------------------------
# Camera setup
# ---------------------------------------------------------------------------

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"Could not open source: {SOURCE}")

# If reading from a webcam, set resolution
if isinstance(SOURCE, int):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

source_fps = cap.get(cv2.CAP_PROP_FPS)
if source_fps == 0 or math.isnan(source_fps):
    source_fps = 30.0

out_writer = None
if OUTPUT_FILE:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, source_fps, (width, height))
    print(f"Saving output to {OUTPUT_FILE} ({width}x{height} @ {source_fps}fps)")

frame_id        = 0
last_detections : list[dict] = []
last_objects    : list[dict] = []


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

try:
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # ------------------------------------------------------------
        # 1. POSE ESTIMATION
        # ------------------------------------------------------------

        pose_result = estimate_pose(frame)
        person      = pose_result["person_pose"]
        named_kps   = pose_result["keypoints"]
        raw_result  = pose_result["raw_result"]

        # ------------------------------------------------------------
        # 2. POSTURE CLASSIFICATION
        # ------------------------------------------------------------

        if person is not None:
            posture, stability = posture_classifier.classify_smoothed(person)
        else:
            posture, stability = "no person detected", 0.0

        # ------------------------------------------------------------
        # 3. OBJECT DETECTION  (every N frames to save compute)
        # ------------------------------------------------------------

        if frame_id % OBJECT_DETECTION_EVERY_N_FRAMES == 0:
            det_result      = detect_objects(frame, conf_threshold=0.15,
                                             imgsz=INFERENCE_IMGSZ)
            last_detections = det_result["detections"]
            _, last_objects = split_persons_and_objects(last_detections)

        objects = last_objects

        # ------------------------------------------------------------
        # 4. RF CLASSIFICATION (per-track affordances)
        # ------------------------------------------------------------
        track_affordances = {}
        if person is not None:
            # Predict using the previous frame's proximity_log for durations
            for det in objects:
                # We need the track_id, but the object hasn't gone through tracker.update() yet.
                # However, tracker.update() is stateful. We can peek at matched tracks via tracker.
                # But to avoid messing with tracker state, we can just run tracker.update() FIRST!
                pass # Fixed below: we'll update tracker manually before calling InteractionEngine
            
            # Safe to update tracker here, because InteractionEngine uses tracks instead of detections.
            # Wait, InteractionEngine accepts `objects` and calls tracker.update(objects).
            # To fix this chicken and egg, we'll let interaction_engine's internal tracker be updated here:
            tracks = interaction_engine.tracker.update(objects)
            for track in tracks:
                affs = _predict_track_affordances(person, track, proximity_log, frame.shape)
                if affs:
                    track_affordances[track.track_id] = affs

        # ------------------------------------------------------------
        # 5. HUMAN-OBJECT INTERACTION  (temporal proximity + duration + CAD affordances)
        # ------------------------------------------------------------

        interaction_result: dict = {"events": [], "proximity_log": [],
                                    "frame_time": 0.0}
        if person is not None:
            # We already updated the tracker manually above. But process_frame() expects `objects` and updates it again!
            # Since update() is deterministic and just updates bounding boxes based on IoU, calling it twice on the SAME objects is perfectly fine.
            # Time Simulation for pre-recorded video files vs Live Webcam
            if isinstance(SOURCE, int):
                simulated_now = time.time()
            else:
                simulated_now = frame_id / source_fps

            interaction_result = interaction_engine.process_frame(
                keypoints   = named_kps,
                objects     = objects,
                bbox_height = max(person.bbox_height, 1.0),
                track_affordances = track_affordances,
                now         = simulated_now,
            )

        interaction_events = interaction_result["events"]
        proximity_log      = interaction_result["proximity_log"]

        rf_label = None
        rf_conf = 0.0

        # ------------------------------------------------------------
        # 6. TRANSPARENT RULE ENGINE  (build explanation)
        # ------------------------------------------------------------

        action, reasons, confidence = build_explanation(
            rf_label           = rf_label,
            rf_confidence      = rf_conf,
            interaction_events = interaction_events,
            proximity_log      = proximity_log,
            posture            = posture,
            track_affordances  = track_affordances,
        )

        # ------------------------------------------------------------
        # 7. DEBUG CONSOLE  (every N frames)
        # ------------------------------------------------------------

        if frame_id % DEBUG_EVERY_N_FRAMES == 0:
            print(f"\n[frame {frame_id}]  posture={posture} ({stability}%)")
            print(f"  objects      = {[o['object'] for o in objects]}")
            print(f"  interaction  = {[e.action for e in interaction_events]}")
            print(f"  RF           = {rf_label}  ({rf_conf}%)")
            print(f"  → displayed  = {action}")

            if person is not None:
                print(f"  knees L={knee_angle(person,'left')}  "
                      f"R={knee_angle(person,'right')}")
                print(f"  torso tilt  = {torso_angle_from_vertical(person)}")

            for rec in proximity_log:
                if rec.duration_sec > 0 or rec.distance_px < 50:
                    print(f"  prox: {rec.keypoint_name:<16} | "
                          f"{rec.object_class:<14} | "
                          f"dist={rec.distance_px:6.1f}px | "
                          f"dur={rec.duration_sec:.2f}s")

        # ------------------------------------------------------------
        # 8. RENDER
        # ------------------------------------------------------------

        annotated = raw_result.plot()   # skeleton overlay from YOLO

        _draw_objects(annotated, last_detections)
        _draw_hud(annotated, posture, stability, action, reasons, confidence)

        if out_writer is not None:
            out_writer.write(annotated)

        cv2.imshow("Project V2C — Human Action Recognition", annotated)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

finally:
    cap.release()
    if out_writer is not None:
        out_writer.release()
    cv2.destroyAllWindows()
    print("\nCamera stopped.")
