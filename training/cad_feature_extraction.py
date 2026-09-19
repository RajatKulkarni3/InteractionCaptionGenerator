"""
Extracts geometric and duration features from the CAD-120 Affordance dataset.
Outputs sequence-level features to outputs/cad_features.csv.
"""

import os
import sys
import csv
from pathlib import Path

import cv2
import numpy as np

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultralytics import YOLO
from src.pose_estimation import estimate_pose

FAR_SENTINEL = 999.0
HAND_NEAR_OBJECT_FRAC = 0.35
MOUTH_NEAR_OBJECT_FRAC = 0.28
CAD_FPS = 30.0  # Assuming 30 fps for CAD 120

AFFORDANCE_NAMES = ["openable", "cuttable", "pourable", "containable", "supportable", "holdable"]

def point_to_box_distance(px, py, x1, y1, x2, y2):
    """Returns 0.0 if point is inside the box, else Euclidean distance to the closest edge/corner."""
    dx = max(0.0, x1 - px, px - x2)
    dy = max(0.0, y1 - py, py - y2)
    return float(np.hypot(dx, dy))

def parse_cad_txt(filepath):
    data = {}
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return data
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            data[parts[0]] = parts[1:]
    return data

def main():
    dataset_root = Path("Dataset/CAD")
    frames_dir = dataset_root / "frames"
    
    crop_info = parse_cad_txt(dataset_root / "crop_coordinate_info.txt")
    affordance_info = parse_cad_txt(dataset_root / "visible_affordance_info.txt")
    
    # Load YOLO11-pose
    pose_model = YOLO("models/yolo11n-pose.pt")
    
    # Group into sequences: contiguous frames for the same track_id with the same affordance
    # Key: track_id (e.g. "_1" -> 1).
    # Since filenames are like 10001_1.png
    instances = []
    for filename, aff_vec in affordance_info.items():
        if filename not in crop_info:
            continue
        try:
            frame_str, track_str = filename.replace(".png", "").split("_")
            frame_id = int(frame_str)
            track_id = int(track_str)
        except ValueError:
            continue
        
        aff_ints = tuple(int(x) for x in aff_vec[:6])
        coords = [float(x) for x in crop_info[filename]]
        instances.append({
            "filename": filename,
            "frame_id": frame_id,
            "track_id": track_id,
            "affordances": aff_ints,
            "coords": coords  # x_center, y_center, width, height
        })
        
    instances.sort(key=lambda x: (x["track_id"], x["frame_id"]))
    
    sequences = []
    current_seq = []
    
    for inst in instances:
        if not current_seq:
            current_seq.append(inst)
        else:
            prev = current_seq[-1]
            if inst["track_id"] == prev["track_id"] and inst["frame_id"] == prev["frame_id"] + 1 and inst["affordances"] == prev["affordances"]:
                current_seq.append(inst)
            else:
                sequences.append(current_seq)
                current_seq = [inst]
    if current_seq:
        sequences.append(current_seq)
        
    print(f"Found {len(sequences)} continuous sequences.")
    
    os.makedirs("outputs", exist_ok=True)
    out_csv = "outputs/cad_features.csv"
    
    # We will compute: mean/min for wrist_dist, face_dist. And max_wrist_near_sec, max_face_near_sec.
    headers = [
        "object_class",
        "mean_wrist_dist", "min_wrist_dist", "max_wrist_near_sec",
        "mean_face_dist", "min_face_dist", "max_face_near_sec",
        "mean_shoulder_dist", "min_shoulder_dist",
        "mean_hip_dist", "min_hip_dist",
        "mean_knee_dist", "min_knee_dist",
        "obj_width_norm", "obj_height_norm",
    ] + AFFORDANCE_NAMES
    
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for i, seq in enumerate(sequences):
            print(f"Processing sequence {i+1}/{len(sequences)} (Frames: {len(seq)})...", end='\r')
            
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
            
            for inst in seq:
                frame_path = frames_dir / f"{inst['frame_id']:05d}_RGB.png"
                if not frame_path.exists():
                    # Fallback to string if 05d padding is not correct
                    base_name = str(inst['filename']).split('_')[0]
                    frame_path = frames_dir / f"{base_name}_RGB.png"
                if not frame_path.exists():
                    continue
                
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    continue
                
                # YOLO Pose
                pose_dict = estimate_pose(frame)
                pose_res = pose_dict["person_pose"]
                kpts = pose_dict["keypoints"]  # dict of named keypoints: {"left_wrist": (x,y), ...}
                
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
                
                # Object bbox (CAD manual bbox)
                # coords: x_center, y_center, width, height
                xc, yc, w, h = inst["coords"]
                x1 = max(0, xc - w / 2)
                y1 = max(0, yc - h / 2)
                x2 = xc + w / 2
                y2 = yc + h / 2
                
                # Wrist dist (min of left and right if available)
                w_dist = FAR_SENTINEL
                for kpt_name in ["left_wrist", "right_wrist"]:
                    if kpt_name in kpts:
                        wx, wy = kpts[kpt_name]
                        d = point_to_box_distance(wx, wy, x1, y1, x2, y2)
                        w_dist = min(w_dist, d / bbox_height)
                
                # Face dist (nose, eyes, ears)
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
                widths_norm.append(w / fw)
                heights_norm.append(h / fh)
                
                if w_dist < HAND_NEAR_OBJECT_FRAC:
                    wrist_streak += 1
                    wrist_best = max(wrist_best, wrist_streak)
                else:
                    wrist_streak = 0
                    
                if f_dist < MOUTH_NEAR_OBJECT_FRAC:
                    face_streak += 1
                    face_best = max(face_best, face_streak)
                else:
                    face_streak = 0
                    
            if not wrist_dists:
                continue
                
            obj_class = seq[0].get("object", "unknown")
            row = [
                obj_class,
                np.mean(wrist_dists),
                np.min(wrist_dists),
                wrist_best / CAD_FPS,
                np.mean(face_dists),
                np.min(face_dists),
                face_best / CAD_FPS,
                np.mean(shoulder_dists),
                np.min(shoulder_dists),
                np.mean(hip_dists),
                np.min(hip_dists),
                np.mean(knee_dists),
                np.min(knee_dists),
                np.mean(widths_norm),
                np.mean(heights_norm),
            ]
            
            # Affordances
            row.extend(seq[0]["affordances"])
            
            writer.writerow(row)
            
    print(f"\nFeature extraction complete. Saved to {out_csv}.")

if __name__ == "__main__":
    main()
