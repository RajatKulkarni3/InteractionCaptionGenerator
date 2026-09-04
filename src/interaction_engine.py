"""
INTERACTION ENGINE — explainable, rule-based human-object interaction detection
================================================================================
Produces output in the shape you asked for:

    Action:  Drinking
    Reason:  Cup detected.
             Right hand moved toward the cup.
             Cup remained close to the mouth for 1.8 seconds.
    Confidence: 94%

WHY RULE-BASED INSTEAD OF A TRAINED HOI MODEL:
A trained model (BLIP captioning, or a real HOI model like QPIC) gives you a
label and maybe a score -- it does NOT hand you "right hand moved toward the
cup" as a justification. To get that reasoning you'd still have to build
explicit logic on top of the model's output. So this engine builds the
reasoning-producing logic directly, using the pose keypoints and object boxes
you already extract, and gets you working structured output today instead of
after a data-collection + training project.

WHAT THIS ADDS ON TOP OF YOUR EXISTING PIPELINE:
1. ObjectTracker: gives each detected object a stable identity across frames
   (matched by class + IoU), so we can measure DURATION ("near mouth for
   1.8s"), not just a single frame's distance. Your current object detection
   has no notion that "this cup" is the same cup 10 frames later.
2. ActionSpec / InteractionEngine: a small, explicit, per-action rule table.
   Each action lists which object classes qualify, which keypoints matter,
   proximity thresholds, and how long a condition must hold. Each condition
   that fires becomes one line of human-readable reasoning -- the reasoning
   list IS the rule trace, not a separate explanation bolted on afterward.
3. Confidence: a transparent formula (proximity margin + duration progress +
   underlying detection confidence), not a learned probability. Same honest
   "how stable is this call" spirit as PostureClassifier.classify_smoothed's
   stability score in pose_estimation_final.py.

HOW TO EXTEND TO MORE ACTIONS:
Add another ActionSpec to ACTION_SPECS below. "drinking" is the fully worked
example; "using phone" is a second, shorter example showing the pattern
generalizes to hand-near-face-region actions without a mouth-duration timer.
Actions that need a totally different shape of evidence (e.g. "sitting on
chair", which is about hip/knee proximity to a chair, not hand/mouth) follow
the same recipe: define which keypoints and object classes matter, define a
proximity threshold, define whether duration matters, and let the reasons
list document whichever conditions actually fired.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------
# Tunable thresholds -- fractions of the person's own bbox_height, the
# same "scale with how close the person is to the camera" trick used
# throughout pose_estimation_final.py, so these don't need re-tuning
# per camera distance.
# ---------------------------------------------------------------------
HAND_NEAR_OBJECT_FRAC = 0.35     # wrist-to-object-center distance below this -> "hand near / picked up"
MOUTH_NEAR_OBJECT_FRAC = 0.28    # face-point-to-object-center distance below this -> "near mouth"
DRINK_MOUTH_DURATION_SEC = 1.5   # object must stay near the mouth at least this long to call it "drinking"
PHONE_NEAR_FACE_DURATION_SEC = 1.0  # phone must stay near hand+face this long to call it "using phone"

TRACK_IOU_MATCH_THRESHOLD = 0.3   # boxes across frames with IoU above this (same class) are the same object
TRACK_MAX_MISSED_FRAMES = 10      # drop a track if it isn't matched for this many consecutive frames


# ---------------------------------------------------------------------
# Object tracking -- gives detections a stable identity across frames
# ---------------------------------------------------------------------

def _iou(box_a, box_b) -> float:
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


@dataclass
class ObjectTrack:
    track_id: int
    cls: str
    bbox: list
    confidence: float
    missed_frames: int = 0
    # Per-condition "since when has this been continuously true" timestamps.
    # None means "not currently true". Keyed loosely by condition name so
    # different actions can each track their own condition without stepping
    # on each other.
    condition_since: dict = field(default_factory=dict)

    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def mark_condition(self, name: str, is_true: bool, now: float) -> float:
        """Updates condition_since[name] and returns how long (seconds) the
        condition has been continuously true (0.0 if not currently true)."""
        if is_true:
            if self.condition_since.get(name) is None:
                self.condition_since[name] = now
            return now - self.condition_since[name]
        else:
            self.condition_since[name] = None
            return 0.0


class ObjectTracker:
    """Simple IoU + class matching tracker. Good enough for a single-camera,
    few-objects scene -- not a general-purpose MOT tracker, just enough to
    give durations meaning instead of re-detecting "a cup" from scratch
    every frame with no memory of the last one."""

    def __init__(self, iou_threshold: float = TRACK_IOU_MATCH_THRESHOLD,
                 max_missed_frames: int = TRACK_MAX_MISSED_FRAMES):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.tracks: dict[int, ObjectTrack] = {}
        self._next_id = 0

    def update(self, detections: list) -> list:
        """detections: list of {"object": cls_name, "confidence": float, "bbox": [x1,y1,x2,y2]}
        Returns the current list of live ObjectTrack objects (matched + new)."""
        unmatched_detections = list(range(len(detections)))
        matched_track_ids = set()

        for track_id, track in self.tracks.items():
            best_idx, best_iou = None, 0.0
            for i in unmatched_detections:
                det = detections[i]
                if det["object"] != track.cls:
                    continue
                iou = _iou(track.bbox, det["bbox"])
                if iou > best_iou:
                    best_idx, best_iou = i, iou
            if best_idx is not None and best_iou >= self.iou_threshold:
                det = detections[best_idx]
                track.bbox = det["bbox"]
                track.confidence = det["confidence"]
                track.missed_frames = 0
                matched_track_ids.add(track_id)
                unmatched_detections.remove(best_idx)

        for track_id, track in self.tracks.items():
            if track_id not in matched_track_ids:
                track.missed_frames += 1

        for i in unmatched_detections:
            det = detections[i]
            track = ObjectTrack(
                track_id=self._next_id,
                cls=det["object"],
                bbox=det["bbox"],
                confidence=det["confidence"],
            )
            self.tracks[self._next_id] = track
            self._next_id += 1

        self.tracks = {
            tid: t for tid, t in self.tracks.items()
            if t.missed_frames <= self.max_missed_frames
        }
        return list(self.tracks.values())


# ---------------------------------------------------------------------
# Interaction result
# ---------------------------------------------------------------------

@dataclass
class InteractionEvent:
    action: str
    reasons: list
    confidence: float  # 0-100

    def format(self) -> str:
        lines = [f"Action:\n{self.action}", "Reason:"]
        lines += [f"{r}" for r in self.reasons]
        lines.append(f"Confidence:\n{round(self.confidence)}%")
        return "\n".join(lines)


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _confidence_from(*components: float) -> float:
    """Simple, transparent average of 0-1 components, expressed as a
    percentage -- not a learned probability, just an honest combination
    of how strong each piece of evidence is."""
    components = [c for c in components if c is not None]
    if not components:
        return 0.0
    return 100.0 * (sum(components) / len(components))


def _margin(distance: float, threshold: float) -> float:
    """0-1 score: 1.0 if distance is 0, 0.0 if distance is at/above
    threshold. Lets 'well within range' score higher than 'barely
    qualifies', instead of every pass/fail check being worth the same."""
    if threshold <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (distance / threshold)))


def _duration_progress(elapsed: float, required: float) -> float:
    return max(0.0, min(1.0, elapsed / required)) if required > 0 else 0.0


# ---------------------------------------------------------------------
# Action detectors
# ---------------------------------------------------------------------
# Each returns an InteractionEvent or None. They're plain functions (not
# a generic config table) so the reasoning text stays specific and
# readable per action -- a fully generic engine tends to produce generic,
# mushy explanations, and the whole point here is specific ones.

DRINK_OBJECT_CLASSES = {"cup", "wine glass", "bottle"}
PHONE_OBJECT_CLASSES = {"cell phone"}


def detect_drinking(track: ObjectTrack, wrist_points: list, face_point: Optional[tuple],
                     bbox_height: float, now: float) -> Optional[InteractionEvent]:
    if track.cls not in DRINK_OBJECT_CLASSES or face_point is None or not wrist_points:
        return None

    obj_center = track.center()

    hand_dist = min(_dist(obj_center, (wx, wy)) for wx, wy, _ in wrist_points)
    hand_near = hand_dist < HAND_NEAR_OBJECT_FRAC * bbox_height
    # which hand, for the reason text
    nearest_wrist_kind = min(wrist_points, key=lambda p: _dist(obj_center, (p[0], p[1])))
    hand_side = "the hand"  # kept generic: kind alone doesn't carry left/right in get_interaction_keypoints

    mouth_dist = _dist(obj_center, face_point)
    mouth_near = mouth_dist < MOUTH_NEAR_OBJECT_FRAC * bbox_height
    mouth_duration = track.mark_condition("near_mouth", mouth_near, now)

    if not (hand_near or mouth_near):
        return None

    reasons = [f"{track.cls.capitalize()} detected."]
    if hand_near:
        reasons.append(f"{hand_side.capitalize()} moved toward the {track.cls}.")
    if mouth_duration > 0:
        reasons.append(f"{track.cls.capitalize()} remained close to the mouth for {mouth_duration:.1f} seconds.")

    if mouth_duration < DRINK_MOUTH_DURATION_SEC:
        return None  # evidence building, but not enough yet to commit to the action

    confidence = _confidence_from(
        _margin(hand_dist, HAND_NEAR_OBJECT_FRAC * bbox_height) if hand_near else 0.5,
        _margin(mouth_dist, MOUTH_NEAR_OBJECT_FRAC * bbox_height),
        _duration_progress(mouth_duration, DRINK_MOUTH_DURATION_SEC),
        track.confidence,
    )
    return InteractionEvent(action="Drinking", reasons=reasons, confidence=confidence)


def detect_using_phone(track: ObjectTrack, wrist_points: list, face_point: Optional[tuple],
                        bbox_height: float, now: float) -> Optional[InteractionEvent]:
    """Second worked example, shorter, to show the pattern generalizes:
    no mouth involved, just hand-near-object AND object-near-face,
    sustained for a shorter duration than drinking (phone-checking is
    typically quicker to confirm than a sustained drink)."""
    if track.cls not in PHONE_OBJECT_CLASSES or face_point is None or not wrist_points:
        return None

    obj_center = track.center()
    hand_dist = min(_dist(obj_center, (wx, wy)) for wx, wy, _ in wrist_points)
    hand_near = hand_dist < HAND_NEAR_OBJECT_FRAC * bbox_height

    face_dist = _dist(obj_center, face_point)
    face_near = face_dist < MOUTH_NEAR_OBJECT_FRAC * bbox_height * 1.5  # a bit more lenient than mouth-only

    both_near = hand_near and face_near
    duration = track.mark_condition("phone_near_hand_and_face", both_near, now)

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
    return InteractionEvent(action="Using phone", reasons=reasons, confidence=confidence)


ACTION_DETECTORS = [detect_drinking, detect_using_phone]


# ---------------------------------------------------------------------
# Top-level engine: call this once per frame
# ---------------------------------------------------------------------

class InteractionEngine:
    def __init__(self):
        self.tracker = ObjectTracker()

    def update(self, detections: list, wrist_points: list, face_point: Optional[tuple],
               bbox_height: float, now: Optional[float] = None) -> list:
        """
        detections: object detector output for this frame, list of
            {"object": cls_name, "confidence": float, "bbox": [x1,y1,x2,y2]}
        wrist_points: list of (x, y, "wrist") from get_interaction_keypoints
            (already filtered to just wrists, or pass the full interaction
            keypoint list -- non-wrist kinds are ignored here)
        face_point: get_face_point(person) output, or None
        bbox_height: person.bbox_height, for scaling thresholds
        now: current time in seconds (time.time() if omitted) -- pass your
            own clock if you want deterministic behaviour on recorded video.

        Returns a list of InteractionEvent, one per action currently firing
        (usually 0 or 1, but nothing stops two different objects from
        triggering different actions in the same frame).
        """
        if now is None:
            now = time.time()

        wrists_only = [p for p in wrist_points if len(p) < 3 or p[2] == "wrist"] or wrist_points

        tracks = self.tracker.update(detections)

        events = []
        for track in tracks:
            for detector in ACTION_DETECTORS:
                event = detector(track, wrists_only, face_point, bbox_height, now)
                if event is not None:
                    events.append(event)
        return events
