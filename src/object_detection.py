"""
OBJECT DETECTION MODULE
========================
Detects objects with YOLOv8n (COCO-pretrained, 80 classes). No class
filtering happens here -- whatever the model detects is returned, so
all 80 COCO classes are available, not a hand-picked subset.

Exposes detect_objects(frame), used by main.py (and previously
imported by it, though the function didn't actually exist in the
version you had -- this restores that interface).
"""

from ultralytics import YOLO

_model = YOLO("yolov8n.pt")


def detect_objects(frame, conf_threshold: float = 0.4, imgsz: int = 640):
    """
    Runs YOLOv8n on a frame.

    imgsz controls the size the frame is resized to before the network
    runs (independent of the frame's actual resolution) -- lower is
    faster, at some cost to detecting small/far-away objects. Defaults
    to Ultralytics' own default (640) so this function's behavior is
    unchanged unless a caller opts into a smaller size.

    Returns a list of dicts:
        "object":     class name (any of the 80 COCO classes)
        "confidence": detection confidence, 0-1
        "bbox":       [x1, y1, x2, y2] in pixel coordinates
    """
    results = _model(frame, verbose=False, conf=conf_threshold, imgsz=imgsz)[0]

    detections = []
    for box in results.boxes:
        detections.append({
            "object": _model.names[int(box.cls)],
            "confidence": float(box.conf),
            "bbox": box.xyxy[0].tolist(),
        })
    return detections


def split_persons_and_objects(detections):
    """
    Splits a detection list into (persons, objects). This is just a
    label split, not a scene judgement -- it exists because the
    relation classifier needs to pair each detected person's box
    against each detected object's box.
    """
    persons = [d for d in detections if d["object"] == "person"]
    objects = [d for d in detections if d["object"] != "person"]
    return persons, objects
