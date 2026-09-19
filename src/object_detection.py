"""
OBJECT DETECTION MODULE — YOLO-World (open-vocabulary)
=======================================================
Replaces YOLOv8n (80 fixed COCO classes) with YOLO-World, which supports
open-vocabulary detection: query any class at inference time by name, not
just the 80 COCO labels baked into a closed-set model.

Weights are loaded exclusively from ``models/yolo_world.pt`` (relative to
the project root). If the file is absent a clear FileNotFoundError is raised
rather than silently downloading a different checkpoint.

Public API
----------
detect_objects(frame, ...) -> dict
    Runs YOLO-World on a frame. Returns a structured dict:
    {
        "detections": [
            {
                "object":     str,            # open-vocabulary class label
                "confidence": float,          # 0-1
                "bbox":       [x1,y1,x2,y2], # pixel coords, float
                "embedding":  list[float],    # semantic feature vector,
                                              # or zero-vector if not exposed
            },
            ...
        ],
        "raw_result": <ultralytics Results>   # for downstream consumers
    }

set_classes(names: list[str]) -> None
    Update the YOLO-World query vocabulary at runtime.  Call this before
    ``detect_objects`` whenever you want to restrict or extend the label set.
    No model reload required.

split_persons_and_objects(detections: list) -> (persons, objects)
    Splits a ``detection_result["detections"]`` list into persons / objects.
    Unchanged contract from the original module.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLOWorld

# ---------------------------------------------------------------------------
# Resolve model path relative to the project root so the module works
# regardless of the working directory the caller uses.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # …/src
_PROJECT_ROOT = _HERE.parent                      # …/InteractionCaptionGenerator
_MODEL_PATH = _PROJECT_ROOT / "models" / "yolo_world.pt"

if not _MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO-World weights not found at '{_MODEL_PATH}'.\n"
        "Download yolo_world.pt (e.g. yolov8s-worldv2.pt from Ultralytics)\n"
        "and place it at that path before running."
    )

_model: YOLOWorld = YOLOWorld(str(_MODEL_PATH))

# Default class vocabulary — mirrors the 80 COCO classes so existing
# interaction rules (cups, phones, …) work out of the box.  Callers can
# call set_classes() to shrink or extend this list at any time.
_DEFAULT_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

_model.set_classes(_DEFAULT_CLASSES)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def set_classes(names: list[str]) -> None:
    """
    Update the YOLO-World query vocabulary at runtime.

    Example::

        set_classes(["cup", "bottle", "cell phone", "person"])

    This does NOT reload the model — YOLO-World embeds the text queries at
    each inference call, so this is effectively free.
    """
    _model.set_classes(names)


def _extract_embedding(result, box_index: int) -> list[float]:
    """
    Extract the semantic embedding for one detection from a YOLO-World result.

    The Ultralytics wrapper does not always expose the per-box text-image
    similarity vector through a stable public attribute, so we try a few
    known attribute paths and fall back to a zero-vector if none is found.
    A future Ultralytics update may change the attribute name; this is the
    correct place to patch it.
    """
    # Attempt 1: direct per-box feature embedding (present in some builds)
    if hasattr(result, "embedding") and result.embedding is not None:
        try:
            emb = result.embedding[box_index]
            return emb.cpu().numpy().tolist()
        except Exception:
            pass

    # Attempt 2: class probability vector as a proxy semantic embedding --
    # shape (num_boxes, num_classes); meaningful for open-vocab similarity.
    if hasattr(result, "boxes") and result.boxes is not None:
        probs = getattr(result.boxes, "cls_prob", None)
        if probs is not None:
            try:
                return probs[box_index].cpu().numpy().tolist()
            except Exception:
                pass

    # Fall-back: zero-vector (number of classes in current vocabulary)
    return [0.0] * len(_DEFAULT_CLASSES)


def detect_objects(
    frame,
    conf_threshold: float = 0.4,
    imgsz: int = 640,
) -> dict:
    """
    Run YOLO-World on *frame* and return a structured detection dict.

    Parameters
    ----------
    frame : np.ndarray
        BGR image as returned by ``cv2.VideoCapture.read()``.
    conf_threshold : float
        Detections below this confidence are discarded (0-1).
    imgsz : int
        Input resolution for the network.  Lower = faster, fewer small
        objects found.  Defaults to Ultralytics' standard 640.

    Returns
    -------
    dict with keys:

    ``"detections"``
        list of dicts, one per detected object::

            {
                "object":     str,             # open-vocabulary class label
                "confidence": float,           # 0-1
                "bbox":       [x1,y1,x2,y2],  # float pixel coords
                "embedding":  list[float],     # semantic feature vector
            }

    ``"raw_result"``
        The raw ``ultralytics.engine.results.Results`` object for
        consumers that need access to masks, keypoints, or annotated plots.
    """
    results = _model(frame, verbose=False, conf=conf_threshold, imgsz=imgsz)
    result = results[0]

    detections: list[dict] = []
    if result.boxes is not None:
        for i, box in enumerate(result.boxes):
            label = _model.names[int(box.cls)]
            detections.append({
                "object":     label,
                "confidence": float(box.conf),
                "bbox":       box.xyxy[0].tolist(),
                "embedding":  _extract_embedding(result, i),
            })

    return {
        "detections": detections,
        "raw_result": result,
    }


def split_persons_and_objects(detections: list) -> tuple[list, list]:
    """
    Split a ``detection_result["detections"]`` list into two lists:
    ``(persons, objects)``  where persons have ``object == "person"``
    and objects are everything else.

    This is a label split, not a scene judgement — it exists so the
    relation classifier can pair each detected person's box against
    each detected object's box independently.
    """
    persons = [d for d in detections if d["object"] == "person"]
    objects = [d for d in detections if d["object"] != "person"]
    return persons, objects
