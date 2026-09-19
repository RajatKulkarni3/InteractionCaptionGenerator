"""
VidHOI Feature Extraction
==========================
Parses VidHOI annotation JSONs (trajectories + relation_instances),
runs YOLO11-Pose on decoded frames, and outputs geometric + duration
features to outputs/vidhoi_features.csv.

Feature schema matches cad_feature_extraction.py:
  mean_wrist_dist, min_wrist_dist, max_wrist_near_sec,
  mean_face_dist, min_face_dist, max_face_near_sec,
  openable, cuttable, pourable, containable, supportable, holdable
"""

import os
import sys
import csv
import json
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pose_estimation import estimate_pose

FAR_SENTINEL = 999.0
SAMPLE_FPS = 5  # Process every Nth frame to reduce compute

# Map VidHOI predicates to our 6 affordance labels
# A relation like "hold bottle" implies the bottle is "holdable"
PREDICATE_TO_AFFORDANCE = {
    "hold":             "holdable",
    "carry":            "holdable",
    "grab":             "holdable",
    "lift":             "holdable",
    "squeeze":          "holdable",
    "pull":             "holdable",
    "push":             "holdable",
    "throw":            "holdable",
    "release":          "holdable",
    "cut":              "cuttable",
    "use":              "openable",   # "use" is ambiguous but often open/manipulate
    "press":            "openable",
    "knock":            "openable",
    "feed":             "pourable",   # feeding often involves pouring/containable items
    "clean":            "containable",
    "lean_on":          "supportable",
    "ride":             "supportable",
    "get_on":           "supportable",
    "play(instrument)": "holdable",
}

# VidHOI categories that map to human subjects (we skip these as objects)
HUMAN_CATEGORIES = {"adult", "child", "baby"}

AFFORDANCE_NAMES = ["openable", "cuttable", "pourable", "containable", "supportable", "holdable"]


def point_to_box_distance(px, py, x1, y1, x2, y2):
    """Returns 0.0 if point is inside the box, else Euclidean distance to closest edge."""
    dx = max(0.0, x1 - px, px - x2)
    dy = max(0.0, y1 - py, py - y2)
    return float(np.hypot(dx, dy))


def main():
    dataset_root = Path("Dataset/VidHOI/extracted")
    frames_root = dataset_root / "frames"
    
    if not frames_root.exists():
        print(f"Error: {frames_root} does not exist. Run setup_vidhoi.py first.")
        return
    
    # Collect all annotation JSONs from both training and validation
    annotation_dirs = []
    for split in ["training", "validation"]:
        split_dir = dataset_root / split
        if split_dir.exists():
            annotation_dirs.append(split_dir)
    
    if not annotation_dirs:
        print("Error: No annotation directories found.")
        return
    
    # Gather all JSON files
    json_files = []
    for ann_dir in annotation_dirs:
        for group_dir in sorted(ann_dir.iterdir()):
            if group_dir.is_dir():
                for jf in sorted(group_dir.glob("*.json")):
                    json_files.append(jf)
                    
    # Cap to first 2000 to prevent multi-hour extraction stalls on huge videos
    json_files = json_files[:2000]
    
    print(f"Found {len(json_files)} annotation files to process.")
    
    os.makedirs("outputs", exist_ok=True)
    out_path = Path("outputs/vidhoi_features.csv")
    
    header = [
        "object_class",
        "mean_wrist_dist", "min_wrist_dist", "max_wrist_near_sec",
        "mean_face_dist", "min_face_dist", "max_face_near_sec",
        "mean_shoulder_dist", "min_shoulder_dist",
        "mean_hip_dist", "min_hip_dist",
        "mean_knee_dist", "min_knee_dist",
        "obj_width_norm", "obj_height_norm",
    ] + AFFORDANCE_NAMES
    
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
    
    total_sequences = 0
    total_written = 0
    
    for ji, json_path in enumerate(json_files):
        print(f"\nProcessing annotation {ji+1}/{len(json_files)}: {json_path.name}...")
        
        with open(json_path) as f:
            annot = json.load(f)
        
        video_id = annot.get("video_id", json_path.stem)
        fps = annot.get("fps", 30.0)
        subjects_objects = annot.get("subject/objects", [])
        trajectories = annot.get("trajectories", [])
        relation_instances = annot.get("relation_instances", [])
        
        # Check if we have decoded frames for this video
        frame_dir = frames_root / str(video_id)
        if not frame_dir.exists():
            continue
        
        # Build tid -> category lookup
        tid_to_category = {}
        for so in subjects_objects:
            tid_to_category[so["tid"]] = so["category"]
        
        # Build frame_id -> {tid: bbox} lookup from trajectories
        # trajectories is a list of lists: trajectories[frame_idx] = [{tid, bbox, ...}, ...]
        frame_bboxes = {}
        for frame_idx, frame_tracks in enumerate(trajectories):
            frame_bboxes[frame_idx] = {}
            for track_entry in frame_tracks:
                tid = track_entry["tid"]
                bb = track_entry["bbox"]
                frame_bboxes[frame_idx][tid] = [bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]]
        
        # Filter relation_instances to only those where subject is human and object is non-human
        # and the predicate maps to one of our affordances
        valid_relations = []
        for ri in relation_instances:
            subj_cat = tid_to_category.get(ri["subject_tid"], "")
            obj_cat = tid_to_category.get(ri["object_tid"], "")
            pred = ri["predicate"]
            
            if subj_cat not in HUMAN_CATEGORIES:
                continue
            if obj_cat in HUMAN_CATEGORIES:
                continue
            if pred not in PREDICATE_TO_AFFORDANCE:
                continue
            
            valid_relations.append(ri)
        
        if not valid_relations:
            continue
        
        total_sequences += len(valid_relations)
        
        # Determine frame sampling: we decoded at 5fps, original videos are at ~30fps
        # So decoded frame 1 = original frame 6, decoded frame 2 = original frame 12, etc.
        decode_fps = 5
        frame_step = max(1, round(fps / decode_fps))
        
        # Get list of available decoded frames
        decoded_frames = sorted(frame_dir.glob("*.jpg"))
        if not decoded_frames:
            continue
        
        # Process each relation instance as a sequence
        for ri_idx, ri in enumerate(valid_relations):
            subj_tid = ri["subject_tid"]
            obj_tid = ri["object_tid"]
            pred = ri["predicate"]
            begin_fid = ri["begin_fid"]
            end_fid = ri["end_fid"]
            
            affordance = PREDICATE_TO_AFFORDANCE[pred]
            aff_vec = [1 if a == affordance else 0 for a in AFFORDANCE_NAMES]
            
            wrist_dists = []
            face_dists = []
            shoulder_dists = []
            hip_dists = []
            knee_dists = []
            widths_norm = []
            heights_norm = []
            wrist_streak = 0
            wrist_best = 0
            face_streak = 0
            face_best = 0
            
            # Sample frames from begin_fid to end_fid
            sampled_fids = list(range(begin_fid, end_fid + 1, frame_step))
            if not sampled_fids:
                continue
            
            for orig_fid in sampled_fids:
                # Map original frame ID to decoded frame index (1-based)
                decoded_idx = (orig_fid // frame_step) + 1
                frame_path = frame_dir / f"{decoded_idx:06d}.jpg"
                
                if not frame_path.exists():
                    continue
                
                # Get object bbox from ground truth trajectories
                if orig_fid not in frame_bboxes:
                    continue
                if obj_tid not in frame_bboxes[orig_fid]:
                    continue
                
                obj_bbox = frame_bboxes[orig_fid][obj_tid]
                x1, y1, x2, y2 = obj_bbox
                
                # Run pose estimation
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    continue
                
                pose_dict = estimate_pose(frame)
                pose_res = pose_dict["person_pose"]
                kpts = pose_dict["keypoints"]
                
                if pose_res is None:
                    wrist_dists.append(FAR_SENTINEL)
                    face_dists.append(FAR_SENTINEL)
                    shoulder_dists.append(FAR_SENTINEL)
                    hip_dists.append(FAR_SENTINEL)
                    knee_dists.append(FAR_SENTINEL)
                    widths_norm.append(0.0)
                    heights_norm.append(0.0)
                    wrist_streak = 0
                    face_streak = 0
                    continue
                
                bbox_height = pose_res.bbox_height
                if bbox_height <= 0:
                    bbox_height = 1.0
                
                # Wrist distance
                w_dist = FAR_SENTINEL
                for kpt_name in ["left_wrist", "right_wrist"]:
                    if kpt_name in kpts:
                        wx, wy = kpts[kpt_name]
                        d = point_to_box_distance(wx, wy, x1, y1, x2, y2)
                        w_dist = min(w_dist, d / bbox_height)
                
                # Face distance
                f_dist = FAR_SENTINEL
                for kpt_name in ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]:
                    if kpt_name in kpts:
                        fx, fy = kpts[kpt_name]
                        d = point_to_box_distance(fx, fy, x1, y1, x2, y2)
                        f_dist = min(f_dist, d / bbox_height)
                
                # Shoulder, Hip, Knee dists
                s_dist = FAR_SENTINEL
                for kpt_name in ["left_shoulder", "right_shoulder"]:
                    if kpt_name in kpts:
                        sx, sy = kpts[kpt_name]
                        d = point_to_box_distance(sx, sy, x1, y1, x2, y2)
                        s_dist = min(s_dist, d / bbox_height)
                        
                h_dist = FAR_SENTINEL
                for kpt_name in ["left_hip", "right_hip"]:
                    if kpt_name in kpts:
                        hx, hy = kpts[kpt_name]
                        d = point_to_box_distance(hx, hy, x1, y1, x2, y2)
                        h_dist = min(h_dist, d / bbox_height)
                        
                k_dist = FAR_SENTINEL
                for kpt_name in ["left_knee", "right_knee"]:
                    if kpt_name in kpts:
                        kx, ky = kpts[kpt_name]
                        d = point_to_box_distance(kx, ky, x1, y1, x2, y2)
                        k_dist = min(k_dist, d / bbox_height)
                        
                wrist_dists.append(w_dist)
                face_dists.append(f_dist)
                shoulder_dists.append(s_dist)
                hip_dists.append(h_dist)
                knee_dists.append(k_dist)
                
                # Size features (normalized by frame size)
                fh, fw = frame.shape[:2]
                widths_norm.append((x2 - x1) / fw)
                heights_norm.append((y2 - y1) / fh)
                
                # Contact streak tracking
                if w_dist < 0.35:
                    wrist_streak += 1
                    wrist_best = max(wrist_best, wrist_streak)
                else:
                    wrist_streak = 0
                
                if f_dist < 0.28:
                    face_streak += 1
                    face_best = max(face_best, face_streak)
                else:
                    face_streak = 0
            
            # Skip sequences with no valid frames
            if not wrist_dists or all(d == FAR_SENTINEL for d in wrist_dists):
                continue
            
            # Compute aggregated features
            w_valid = [d for d in wrist_dists if d != FAR_SENTINEL]
            f_valid = [d for d in face_dists if d != FAR_SENTINEL]
            s_valid = [d for d in shoulder_dists if d != FAR_SENTINEL]
            h_valid = [d for d in hip_dists if d != FAR_SENTINEL]
            k_valid = [d for d in knee_dists if d != FAR_SENTINEL]
            
            mean_w = sum(w_valid) / len(w_valid) if w_valid else FAR_SENTINEL
            min_w = min(w_valid) if w_valid else FAR_SENTINEL
            max_wrist_sec = wrist_best * (frame_step / fps)
            
            mean_f = sum(f_valid) / len(f_valid) if f_valid else FAR_SENTINEL
            min_f = min(f_valid) if f_valid else FAR_SENTINEL
            max_face_sec = face_best * (frame_step / fps)
            
            mean_s = sum(s_valid) / len(s_valid) if s_valid else FAR_SENTINEL
            min_s = min(s_valid) if s_valid else FAR_SENTINEL
            
            mean_h = sum(h_valid) / len(h_valid) if h_valid else FAR_SENTINEL
            min_h = min(h_valid) if h_valid else FAR_SENTINEL
            
            mean_k = sum(k_valid) / len(k_valid) if k_valid else FAR_SENTINEL
            min_k = min(k_valid) if k_valid else FAR_SENTINEL
            
            mean_width = sum(widths_norm) / len(widths_norm) if widths_norm else 0.0
            mean_height = sum(heights_norm) / len(heights_norm) if heights_norm else 0.0
            
            obj_class = tid_to_category.get(obj_tid, "unknown")
            
            row = [
                obj_class,
                mean_w, min_w, max_wrist_sec,
                mean_f, min_f, max_face_sec,
                mean_s, min_s,
                mean_h, min_h,
                mean_k, min_k,
                mean_width, mean_height
            ] + aff_vec
            
            with open(out_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            total_written += 1
        
        print(f"  Relations processed: {len(valid_relations)}, written so far: {total_written}")
    
    print(f"\nFeature extraction complete.")
    print(f"Total sequences found: {total_sequences}")
    print(f"Total rows written: {total_written}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
