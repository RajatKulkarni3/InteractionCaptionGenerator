"""
INTERACTION ENGINE — continuous-video, keypoint-to-object proximity tracking
=============================================================================
Processes sequential video frames to measure *spatial distance* between human
keypoints (from YOLO11-Pose) and object bounding boxes (from YOLO-World), and
accumulates *temporal duration* for any keypoint that intersects or remains
close to a box across successive frames.

Architecture overview
---------------------

┌─────────────────────────────────────────────────────────────────────┐
│ Per-frame input                                                     │
│   keypoints : dict[str, (x,y)]  — from estimate_pose()["keypoints"]│
│   objects   : list[det dict]    — from detect_objects()["detections"]│
│   bbox_height: float            — person bbox height (for scaling) │
│   now        : float            — current wall-clock time (seconds)│
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  ObjectTracker.update()      │  gives each object a stable
              │  (IoU + class matching)      │  identity across frames
              └──────────────┬──────────────┘
                             │
     ┌───────────────────────▼───────────────────────┐
     │  calculate_keypoint_object_distances()         │
     │  for every (keypoint, tracked-object) pair:   │
     │    distance = point-to-box distance (px)      │
     │    0.0 when point is INSIDE the box           │
     └───────────────────────┬───────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  DurationTracker.update()    │  accumulates continuous
              │  per (keypoint, track_id)    │  seconds of intersection
              └──────────────┬──────────────┘
                             │
     ┌───────────────────────▼───────────────────────┐
     │  Action detectors (detect_drinking, …)        │
     │  same rule-based detectors as before, but now │
     │  fed from the unified proximity log           │
     └───────────────────────┬───────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  process_frame() output dict │
              │   "events"       : list[…]  │
              │   "proximity_log": list[…]  │
              │   "frame_time"   : float    │
              └─────────────────────────────┘

Key design decisions
--------------------
* ``_point_to_box_distance`` returns the **minimum distance from the point to
  the nearest point on the box boundary** (0.0 when inside) rather than
  center-to-center distance.  This matches the geometric intuition of
  "touching": a wrist keypoint that sits inside the cup's bounding box has
  zero distance, not some non-zero center-to-center number.

* ``DurationTracker`` tracks per ``(keypoint_name, track_id)`` pairs.  Each
  pair independently starts its own timer the first frame the point enters the
  box, and resets to 0.0 the frame the point leaves.  The running elapsed
  time is a plain Python ``float`` (seconds since first contact).

* All existing components (``ObjectTracker``, ``InteractionEvent``,
  ``detect_drinking``, ``detect_using_phone``) are preserved and wired to the
  new output format.  ``main.py`` reads ``result["events"]`` instead of the
  bare list returned by the old ``update()``; the list contents are identical.

* ``InteractionEngine.update()`` is kept as a backward-compatible wrapper that
  delegates to ``process_frame()`` and returns just the events list, so any
  caller that hasn't been updated yet still works.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Tunable thresholds — fractions of the person's own bbox_height so they scale
# with camera distance (the same convention used throughout pose_estimation.py)
# ─────────────────────────────────────────────────────────────────────────────

HAND_NEAR_OBJECT_FRAC    = 0.35   # wrist point-to-box distance < this → "near / holding"
MOUTH_NEAR_OBJECT_FRAC   = 0.28   # face point-to-box distance  < this → "near mouth"
DRINK_MOUTH_DURATION_SEC = 1.5    # object must stay near mouth this long → "drinking"
PHONE_NEAR_FACE_DURATION_SEC = 1.0  # phone near hand+face this long → "using phone"
LAPTOP_WRIST_DURATION_SEC = 2.0   # wrist near laptop this long → "using laptop"
READING_DURATION_SEC      = 2.0   # hand + face near book this long → "reading"

# Grace period: DurationTracker freezes the timer for this many consecutive
# frames of non-contact before resetting it.  YOLO and pose estimators naturally
# flicker — a single missed frame must NOT erase 2 seconds of accumulated evidence.
DURATION_GRACE_FRAMES = 10

TRACK_IOU_MATCH_THRESHOLD = 0.3   # IoU threshold for matching boxes across frames
TRACK_MAX_MISSED_FRAMES   = 10    # drop a track after this many consecutive misses

# Keypoints whose proximity to objects is always recorded in the proximity log
_INTERACTION_KEYPOINTS = {
    "left_wrist", "right_wrist",
    "left_elbow", "right_elbow",
    "left_hip",   "right_hip",
    "left_knee",  "right_knee",
    "nose",
}


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────

def _iou(box_a: list, box_b: list) -> float:
    """Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _point_to_box_distance(pt: tuple[float, float], box: list[float]) -> float:
    """
    Minimum Euclidean distance from point *pt* = (x, y) to the nearest point
    on (or inside) the axis-aligned bounding *box* = [x1, y1, x2, y2].

    Returns 0.0 when the point lies inside the box — meaning a keypoint that
    is *inside* the object's bounding box reports zero distance, which is the
    correct signal for "the hand is touching / holding the object".

    This is intentionally NOT center-to-center distance: a wrist at the edge
    of a large cup's box sits far from its center even though it is geometrically
    touching the cup's rim, which would produce a misleadingly large number.
    """
    px, py = pt
    x1, y1, x2, y2 = box

    # Clamp the point to the box edges (gives 0 when inside)
    cx = max(x1, min(px, x2))
    cy = max(y1, min(py, y2))
    return math.hypot(px - cx, py - cy)


# ─────────────────────────────────────────────────────────────────────────────
# Object tracking — stable identities across frames
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ObjectTrack:
    """A tracked object with a stable ``track_id`` across video frames."""
    track_id: int
    cls: str
    bbox: list
    confidence: float
    missed_frames: int = 0
    # Per-condition timers keyed by condition name — None = not currently active.
    # Different action detectors each track their own condition independently.
    condition_since: dict = field(default_factory=dict)

    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def mark_condition(self, name: str, is_true: bool, now: float) -> float:
        """
        Update the timer for *name* and return how many continuous seconds it
        has been True (0.0 if currently False or just became True this frame).
        """
        if is_true:
            if self.condition_since.get(name) is None:
                self.condition_since[name] = now
            return now - self.condition_since[name]
        else:
            self.condition_since[name] = None
            return 0.0


class ObjectTracker:
    """
    Simple IoU + class-label tracker.  Good enough for a single-camera,
    few-objects scene — not a full MOT tracker, just enough to give each
    detection a stable ``track_id`` so durations are meaningful.
    """

    def __init__(
        self,
        iou_threshold: float = TRACK_IOU_MATCH_THRESHOLD,
        max_missed_frames: int = TRACK_MAX_MISSED_FRAMES,
    ):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.tracks: dict[int, ObjectTrack] = {}
        self._next_id = 0

    def update(self, detections: list[dict]) -> list[ObjectTrack]:
        """
        Match *detections* (list of ``{"object", "confidence", "bbox", …}`` dicts)
        against existing tracks and return the current live ``ObjectTrack`` list.
        """
        unmatched = list(range(len(detections)))
        matched_ids: set[int] = set()

        for track_id, track in self.tracks.items():
            best_idx, best_iou = None, 0.0
            for i in unmatched:
                det = detections[i]
                if det["object"] != track.cls:
                    continue
                iou = _iou(track.bbox, det["bbox"])
                if iou > best_iou:
                    best_idx, best_iou = i, iou
            if best_idx is not None and best_iou >= self.iou_threshold:
                det = detections[best_idx]
                track.bbox       = det["bbox"]
                track.confidence = det["confidence"]
                track.missed_frames = 0
                matched_ids.add(track_id)
                unmatched.remove(best_idx)

        for track_id, track in self.tracks.items():
            if track_id not in matched_ids:
                track.missed_frames += 1

        for i in unmatched:
            det = detections[i]
            self.tracks[self._next_id] = ObjectTrack(
                track_id=self._next_id,
                cls=det["object"],
                bbox=det["bbox"],
                confidence=det["confidence"],
            )
            self._next_id += 1

        self.tracks = {
            tid: t for tid, t in self.tracks.items()
            if t.missed_frames <= self.max_missed_frames
        }
        return list(self.tracks.values())


# ─────────────────────────────────────────────────────────────────────────────
# Duration tracking — per (keypoint_name, track_id) intersection timers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProximityRecord:
    """
    Spatial + temporal relationship between one keypoint and one tracked object
    for a single video frame.

    ``duration_sec`` is the *continuous* number of seconds this keypoint has
    intersected the object's bounding box without interruption.  It is a plain
    ``float`` (e.g. 2.34) that resets to 0.0 the first frame the keypoint
    leaves the box.  A non-zero value means "has been inside for this long".
    """
    keypoint_name:  str
    track_id:       int
    object_class:   str
    distance_px:    float   # point-to-box distance; 0.0 when inside the box
    duration_sec:   float   # continuous seconds of intersection (0.0 = not inside)
    frame_time:     float   # wall-clock time of this frame (seconds)


class DurationTracker:
    """
    Accumulates *continuous* intersection durations per ``(keypoint_name, track_id)``
    pair across video frames.

    Internally stores ``start_time`` the first frame the distance is 0.0
    (keypoint inside box) and computes elapsed = now - start_time.

    Grace period
    ------------
    YOLO detection and pose estimation naturally flicker — a keypoint that is
    genuinely inside a box may read as 1–2 px outside for one or two frames
    due to jitter.  Without a grace period this would reset a 2-second timer
    to 0.0, preventing drinking/phone detectors from ever firing.

    When ``distance_px > 0`` the miss counter increments each frame.  The
    timer is *frozen* (not reset) while ``miss_count <= DURATION_GRACE_FRAMES``.
    Only after ``DURATION_GRACE_FRAMES`` consecutive non-contact frames is the
    timer cleared.  This makes timers robust to single-frame dropout without
    making them tolerant of genuine contact breaks.
    """

    def __init__(self) -> None:
        # (keypoint_name, track_id) -> start_time (float) or None
        self._active: dict[tuple[str, int], Optional[float]] = {}
        # consecutive frames since last contact (0 = currently inside)
        self._miss_count: dict[tuple[str, int], int] = {}

    def update(
        self,
        keypoint_name: str,
        track_id: int,
        distance_px: float,
        now: float,
    ) -> float:
        """
        Update the timer for *(keypoint_name, track_id)* and return the
        continuously accumulated intersection duration in seconds.

        Parameters
        ----------
        distance_px : float
            Current point-to-box distance.  0.0 means the keypoint is
            inside the bounding box (intersection active).
        now : float
            Current wall-clock time in seconds (``time.time()``).

        Returns
        -------
        float
            Seconds the keypoint has been continuously (or recently) inside
            this object's box.  0.0 only after DURATION_GRACE_FRAMES
            consecutive non-contact frames.
        """
        key = (keypoint_name, track_id)
        inside = distance_px == 0.0

        if inside:
            self._miss_count[key] = 0
            if self._active.get(key) is None:
                self._active[key] = now   # first frame inside — start timer
            return now - self._active[key]
        else:
            miss = self._miss_count.get(key, 0) + 1
            self._miss_count[key] = miss
            if miss <= DURATION_GRACE_FRAMES and self._active.get(key) is not None:
                # Within grace period — freeze the timer instead of resetting
                return now - self._active[key]
            # Grace period expired — genuine contact break
            self._active[key] = None
            return 0.0

    def cleanup_stale_tracks(self, live_track_ids: set[int]) -> None:
        """Remove timers for tracks that the ObjectTracker has dropped."""
        stale = [k for k in self._active if k[1] not in live_track_ids]
        for k in stale:
            del self._active[k]
            self._miss_count.pop(k, None)


# ─────────────────────────────────────────────────────────────────────────────
# Proximity calculation (pure function — no side effects, easy to unit-test)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_keypoint_object_distances(
    keypoints: dict[str, tuple[float, float]],
    tracks: list[ObjectTrack],
    bbox_height: float,
) -> list[tuple[str, ObjectTrack, float]]:
    """
    Compute the point-to-box distance between every visible keypoint and
    every tracked object.

    Parameters
    ----------
    keypoints : dict[str, (x, y)]
        Named keypoint positions (already filtered to ≥ 0.5 confidence by
        ``estimate_pose()``).
    tracks : list[ObjectTrack]
        Current live tracks from ``ObjectTracker.update()``.
    bbox_height : float
        Person bounding-box height in pixels (for optional future
        relative-distance calculations by callers; not used internally here).

    Returns
    -------
    list of ``(keypoint_name, track, distance_px)`` triples — one entry per
    (keypoint × object) pair where both are present.  Distance is 0.0 when
    the keypoint is inside the object's bounding box.
    """
    pairs: list[tuple[str, ObjectTrack, float]] = []
    for kp_name, (kx, ky) in keypoints.items():
        for track in tracks:
            dist = _point_to_box_distance((kx, ky), track.bbox)
            pairs.append((kp_name, track, dist))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Interaction result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InteractionEvent:
    """A detected human-object interaction with structured reasoning."""
    action: str
    object_name: str
    reasons: list[str]
    confidence: float  # 0–100

    def format(self) -> str:
        lines = [f"Action:\n{self.action}", "Reason:"]
        lines += [str(r) for r in self.reasons]
        lines.append(f"Confidence:\n{round(self.confidence)}%")
        return "\n".join(lines)


def _dist_pts(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _confidence_from(*components: float) -> float:
    """Transparent average of 0–1 components, as a percentage."""
    comps = [c for c in components if c is not None]
    if not comps:
        return 0.0
    return 100.0 * (sum(comps) / len(comps))


def _margin(distance: float, threshold: float) -> float:
    """0–1 score: 1.0 when distance=0, 0.0 at/above threshold."""
    if threshold <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (distance / threshold)))


def _duration_progress(elapsed: float, required: float) -> float:
    return max(0.0, min(1.0, elapsed / required)) if required > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Action detectors
# ─────────────────────────────────────────────────────────────────────────────

DRINK_OBJECT_CLASSES  = {"cup", "wine glass", "bottle"}
PHONE_OBJECT_CLASSES  = {"cell phone"}
LAPTOP_OBJECT_CLASSES = {"laptop"}
READING_OBJECT_CLASSES = {"book"}


def detect_drinking(
    track: ObjectTrack,
    wrist_points: list,
    face_point: Optional[tuple],
    bbox_height: float,
    now: float,
) -> Optional[InteractionEvent]:
    if track.cls not in DRINK_OBJECT_CLASSES or face_point is None or not wrist_points:
        return None

    obj_center = track.center()
    hand_dist  = min(_dist_pts(obj_center, (wx, wy)) for wx, wy, *_ in wrist_points)
    hand_near  = hand_dist < HAND_NEAR_OBJECT_FRAC * bbox_height

    mouth_dist     = _dist_pts(obj_center, face_point)
    mouth_near     = mouth_dist < MOUTH_NEAR_OBJECT_FRAC * bbox_height
    mouth_duration = track.mark_condition("near_mouth", mouth_near, now)

    if not (hand_near or mouth_near):
        return None

    reasons = [f"{track.cls.capitalize()} detected."]
    if hand_near:
        reasons.append(f"Hand moved toward the {track.cls}.")
    if mouth_duration > 0:
        reasons.append(
            f"{track.cls.capitalize()} remained close to the mouth "
            f"for {mouth_duration:.1f} seconds."
        )

    if mouth_duration < DRINK_MOUTH_DURATION_SEC:
        return None  # still building evidence

    confidence = _confidence_from(
        _margin(hand_dist,  HAND_NEAR_OBJECT_FRAC * bbox_height) if hand_near else 0.5,
        _margin(mouth_dist, MOUTH_NEAR_OBJECT_FRAC * bbox_height),
        _duration_progress(mouth_duration, DRINK_MOUTH_DURATION_SEC),
        track.confidence,
    )
    return InteractionEvent(action="drinking", object_name=track.cls, reasons=reasons, confidence=confidence)


def detect_using_phone(
    track: ObjectTrack,
    wrist_points: list,
    face_point: Optional[tuple],
    bbox_height: float,
    now: float,
) -> Optional[InteractionEvent]:
    if track.cls not in PHONE_OBJECT_CLASSES or face_point is None or not wrist_points:
        return None

    obj_center = track.center()
    hand_dist  = min(_dist_pts(obj_center, (wx, wy)) for wx, wy, *_ in wrist_points)
    hand_near  = hand_dist < HAND_NEAR_OBJECT_FRAC * bbox_height

    face_dist = _dist_pts(obj_center, face_point)
    face_near = face_dist < MOUTH_NEAR_OBJECT_FRAC * bbox_height * 1.5

    both_near = hand_near and face_near
    duration  = track.mark_condition("phone_near_hand_and_face", both_near, now)

    if duration < PHONE_NEAR_FACE_DURATION_SEC:
        return None

    reasons = [
        "Phone detected.",
        "Hand moved toward the phone.",
        f"Phone remained near the hand and face for {duration:.1f} seconds.",
    ]
    confidence = _confidence_from(
        _margin(hand_dist, HAND_NEAR_OBJECT_FRAC * bbox_height),
        _margin(face_dist, MOUTH_NEAR_OBJECT_FRAC * bbox_height * 1.5),
        _duration_progress(duration, PHONE_NEAR_FACE_DURATION_SEC),
        track.confidence,
    )
    return InteractionEvent(action="using", object_name=track.cls, reasons=reasons, confidence=confidence)


def detect_using_laptop(
    track: ObjectTrack,
    wrist_points: list,
    face_point: Optional[tuple],
    bbox_height: float,
    now: float,
) -> Optional[InteractionEvent]:
    """
    Fires when a wrist has been near the laptop for at least
    ``LAPTOP_WRIST_DURATION_SEC`` seconds.  No face proximity required —
    laptop use is primarily a hand activity (typing / trackpad).
    An optional face-toward-screen bonus raises confidence.
    """
    if track.cls not in LAPTOP_OBJECT_CLASSES or not wrist_points:
        return None

    obj_center = track.center()
    hand_dist  = min(_dist_pts(obj_center, (wx, wy)) for wx, wy, *_ in wrist_points)
    hand_near  = hand_dist < HAND_NEAR_OBJECT_FRAC * bbox_height

    duration = track.mark_condition("wrist_near_laptop", hand_near, now)

    if duration < LAPTOP_WRIST_DURATION_SEC:
        return None

    reasons = [
        "Laptop detected.",
        f"Hand near the laptop for {duration:.1f} seconds.",
    ]

    face_component = 0.5   # neutral if face point not available
    if face_point is not None:
        face_dist = _dist_pts(obj_center, face_point)
        # Laptop screen is typically 1–2 person-heights away from the face
        face_near = face_dist < MOUTH_NEAR_OBJECT_FRAC * bbox_height * 3.0
        if face_near:
            reasons.append("Face oriented toward the screen.")
        face_component = _margin(face_dist, MOUTH_NEAR_OBJECT_FRAC * bbox_height * 3.0)

    confidence = _confidence_from(
        _margin(hand_dist, HAND_NEAR_OBJECT_FRAC * bbox_height),
        _duration_progress(duration, LAPTOP_WRIST_DURATION_SEC),
        face_component,
        track.confidence,
    )
    return InteractionEvent(action="using", object_name=track.cls, reasons=reasons, confidence=confidence)


def detect_reading(
    track: ObjectTrack,
    wrist_points: list,
    face_point: Optional[tuple],
    bbox_height: float,
    now: float,
) -> Optional[InteractionEvent]:
    """
    Fires when a book has been near both a wrist AND the face for at least
    ``READING_DURATION_SEC`` seconds.  Both proximity conditions are required
    because a book lying on a table satisfies the wrist condition alone;
    bringing it up in front of the face is the discriminating signal.
    """
    if track.cls not in READING_OBJECT_CLASSES or face_point is None or not wrist_points:
        return None

    obj_center = track.center()
    hand_dist  = min(_dist_pts(obj_center, (wx, wy)) for wx, wy, *_ in wrist_points)
    hand_near  = hand_dist < HAND_NEAR_OBJECT_FRAC * bbox_height

    face_dist = _dist_pts(obj_center, face_point)
    # Books are held at arm's length — allow a larger face proximity radius
    face_near = face_dist < MOUTH_NEAR_OBJECT_FRAC * bbox_height * 2.5

    both_near = hand_near and face_near
    duration  = track.mark_condition("reading", both_near, now)

    if duration < READING_DURATION_SEC:
        return None

    reasons = [
        "Book detected.",
        f"Book held in hand for {duration:.1f} seconds.",
        "Face oriented toward the book.",
    ]
    confidence = _confidence_from(
        _margin(hand_dist, HAND_NEAR_OBJECT_FRAC * bbox_height),
        _margin(face_dist, MOUTH_NEAR_OBJECT_FRAC * bbox_height * 2.5),
        _duration_progress(duration, READING_DURATION_SEC),
        track.confidence,
    )
    return InteractionEvent(action="reading", object_name=track.cls, reasons=reasons, confidence=confidence)


ACTION_DETECTORS = [
    detect_drinking,
    detect_using_phone,
    detect_using_laptop,
    detect_reading,
]

AFFORDANCE_ACTION_MAP = {
    "pourable": {"action": "Pouring", "min_duration": 1.5, "present_tense": "pouring"},
    "openable": {"action": "Opening", "min_duration": 1.0, "present_tense": "opening"},
    "cuttable": {"action": "Cutting", "min_duration": 1.5, "present_tense": "cutting"},
    "containable": {"action": "Filling/Containing", "min_duration": 1.0, "present_tense": "containing"},
    "supportable": {"action": "Supporting", "min_duration": 1.0, "present_tense": "supporting"},
    "holdable": {"action": "Holding", "min_duration": 0.5, "present_tense": "holding"},
}

def detect_affordance_based_action(
    track: ObjectTrack,
    predicted_affordances: list[str],
    wrist_points: list,
    bbox_height: float,
    now: float,
) -> Optional[InteractionEvent]:
    # Defense-in-depth: the CAD affordance model's object_class feature was
    # trained with every row labeled "unknown" (see
    # training/cad_feature_extraction.py), so it cannot reliably tell object
    # identities apart and should never override a class that already has a
    # dedicated, tested rule detector (drinking/using phone/using laptop/
    # reading). The caller (main.py) already avoids computing affordances
    # for these classes; this check makes that guarantee hold even if
    # predicted_affordances is populated by some other caller.
    if track.cls in (DRINK_OBJECT_CLASSES | PHONE_OBJECT_CLASSES
                      | LAPTOP_OBJECT_CLASSES | READING_OBJECT_CLASSES):
        return None
    if not predicted_affordances or not wrist_points:
        return None

    obj_center = track.center()
    hand_dist  = min(_dist_pts(obj_center, (wx, wy)) for wx, wy, *_ in wrist_points)
    hand_near  = hand_dist < HAND_NEAR_OBJECT_FRAC * bbox_height

    duration = track.mark_condition("wrist_near_affordance_obj", hand_near, now)

    # Pick the most "active" affordance (priority: pourable, cuttable, openable over holdable)
    best_aff = None
    best_conf = 0.0
    
    for aff in ["pourable", "cuttable", "openable", "containable", "supportable", "holdable"]:
        if aff in predicted_affordances:
            rule = AFFORDANCE_ACTION_MAP[aff]
            if duration >= rule["min_duration"]:
                best_aff = aff
                break

    if best_aff is None:
        return None
        
    rule = AFFORDANCE_ACTION_MAP[best_aff]
    reasons = [
        f"{track.cls.capitalize()} detected with '{best_aff}' affordance.",
        f"Hand near object for {duration:.1f} seconds.",
    ]
    confidence = _confidence_from(
        _margin(hand_dist, HAND_NEAR_OBJECT_FRAC * bbox_height),
        _duration_progress(duration, rule["min_duration"]),
        track.confidence,
    )
    
    return InteractionEvent(action=rule["action"], reasons=reasons, confidence=confidence)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level engine
# ─────────────────────────────────────────────────────────────────────────────

class InteractionEngine:
    """
    Stateful engine that processes continuous video frames.

    Usage (per-frame)::

        engine = InteractionEngine()

        # inside your frame loop:
        result = engine.process_frame(
            keypoints  = pose_result["keypoints"],    # dict[str, (x,y)]
            objects    = detection_result["detections"],
            bbox_height= person.bbox_height,
            now        = time.time(),
        )

        for event in result["events"]:
            print(event.format())

        for prox in result["proximity_log"]:
            # prox.duration_sec is a float: continuous seconds of intersection
            print(prox.keypoint_name, prox.object_class, prox.duration_sec)
    """

    def __init__(self) -> None:
        self.tracker          = ObjectTracker()
        self.duration_tracker = DurationTracker()

    def process_frame(
        self,
        keypoints:   dict[str, tuple[float, float]],
        objects:     list[dict],
        bbox_height: float,
        track_affordances: dict[int, list[str]] = None,
        now:         Optional[float] = None,
    ) -> dict:
        """
        Process one video frame of pose + object detection data.

        Parameters
        ----------
        keypoints : dict[str, (x, y)]
            Named keypoint positions from ``estimate_pose()["keypoints"]``.
            Only keypoints already passing the 0.5 confidence threshold are
            expected here (``estimate_pose`` filters them upstream).
        objects : list[dict]
            Object detection dicts from ``detect_objects()["detections"]``:
            each must have keys ``"object"``, ``"confidence"``, ``"bbox"``.
        bbox_height : float
            Person bounding-box height in pixels, used to scale proximity
            thresholds with camera distance.
        now : float, optional
            Wall-clock time in seconds.  Defaults to ``time.time()``.
            Pass an explicit value for deterministic replay on recorded video.

        Returns
        -------
        dict with keys:

        ``"events"``
            ``list[InteractionEvent]`` — actions currently firing
            (typically 0 or 1).

        ``"proximity_log"``
            ``list[ProximityRecord]`` — one record per (keypoint × object)
            pair that was evaluated this frame.  Each record exposes:
            - ``distance_px``  : float, 0.0 when the keypoint is inside the box
            - ``duration_sec`` : float, continuously accumulated intersection
              time in seconds; resets to 0.0 when the keypoint leaves the box

        ``"frame_time"``
            ``float`` — the wall-clock timestamp of this frame.
        """
        if now is None:
            now = time.time()

        # 1. Update object tracks
        tracks = self.tracker.update(objects)
        live_ids = {t.track_id for t in tracks}
        self.duration_tracker.cleanup_stale_tracks(live_ids)

        # 2. Compute all (keypoint, object) distances
        raw_pairs = calculate_keypoint_object_distances(keypoints, tracks, bbox_height)

        # 3. Build proximity log, updating DurationTracker for each pair
        proximity_log: list[ProximityRecord] = []
        for kp_name, track, dist in raw_pairs:
            duration = self.duration_tracker.update(kp_name, track.track_id, dist, now)
            proximity_log.append(ProximityRecord(
                keypoint_name=kp_name,
                track_id=track.track_id,
                object_class=track.cls,
                distance_px=dist,
                duration_sec=duration,
                frame_time=now,
            ))

        # 4. Build wrist_points and face_point in the shape that action
        #    detectors expect, sourcing from the already-filtered keypoints dict.
        wrist_points: list = []
        for side in ("left_wrist", "right_wrist"):
            if side in keypoints:
                x, y = keypoints[side]
                wrist_points.append((x, y, "wrist"))

        face_point: Optional[tuple] = None
        for kp_name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
            if kp_name in keypoints:
                face_point = keypoints[kp_name]
                break

        # 5. Run action detectors
        events: list[InteractionEvent] = []
        if track_affordances is None:
            track_affordances = {}
            
        for track in tracks:
            # Traditional detectors
            for detector in ACTION_DETECTORS:
                event = detector(track, wrist_points, face_point, bbox_height, now)
                if event is not None:
                    events.append(event)
            
            # CAD-based affordance detector
            affs = track_affordances.get(track.track_id, [])
            if affs:
                aff_event = detect_affordance_based_action(track, affs, wrist_points, bbox_height, now)
                if aff_event is not None:
                    events.append(aff_event)

        return {
            "events":        events,
            "proximity_log": proximity_log,
            "frame_time":    now,
        }

    # ------------------------------------------------------------------
    # Backward-compatible wrapper for callers that haven't been updated
    # ------------------------------------------------------------------

    def update(
        self,
        detections:  list,
        wrist_points: list,
        face_point:  Optional[tuple],
        bbox_height: float,
        now:         Optional[float] = None,
    ) -> list[InteractionEvent]:
        """
        Legacy interface kept for ``main.py`` compatibility.

        Converts the old ``(detections, wrist_points, face_point, bbox_height)``
        call signature into a ``process_frame()`` call by reconstructing the
        keypoints dict from the provided wrist_points and face_point.

        For new code, call ``process_frame()`` directly with the full named
        keypoints dict from ``estimate_pose()``.
        """
        if now is None:
            now = time.time()

        # Reconstruct minimal keypoints dict from the legacy arguments
        keypoints: dict[str, tuple[float, float]] = {}
        for pt in wrist_points:
            x, y = pt[0], pt[1]
            kind = pt[2] if len(pt) > 2 else "wrist"
            # We can't recover left/right here, so assign generically
            key = "left_wrist" if "left_wrist" not in keypoints else "right_wrist"
            keypoints[key] = (x, y)
        if face_point is not None:
            keypoints["nose"] = face_point

        result = self.process_frame(
            keypoints=keypoints,
            objects=detections,
            bbox_height=bbox_height,
            now=now,
        )
        return result["events"]
