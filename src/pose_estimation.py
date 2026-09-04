"""
POSE ESTIMATION MODULE — Human Action Recognition Internship
================================================================
Scope: pose estimation + posture classification ONLY.
Object detection / human-object-interaction is a separate teammate's
component, to be integrated downstream later. This file has zero
dependency on any object detector.

MODEL: YOLO11-Pose (Ultralytics), pretrained on COCO.
  Academic lineage: Maji, D., Nagori, S., Mathew, M., Poddar, D.
  "YOLO-Pose: Enhancing YOLO for Multi-Person Pose Estimation Using
  Object Keypoint Similarity Loss." CVPR Workshops, 2022. arXiv:2204.06806.
  (Note: the specific pretrained checkpoint used here, via Ultralytics,
  is an engineering implementation of this paradigm and does not have
  its own peer-reviewed paper -- stated explicitly for transparency.)

------------------------------------------------------------------
CHANGELOG
------------------------------------------------------------------
v5 (this version):
  - FIXED "standing" being unreachable whenever ankles aren't visible
    on either leg. Root cause: knee_angle() -- and therefore the whole
    standing/sitting/crouching decision below it -- needs hip + knee +
    ankle ALL visible per leg. When shins/ankles are cropped out of a
    webcam frame or hidden under a desk but the THIGHS (hip+knee) are
    still visible, that used to fail exactly like the "no legs at all"
    case and fall straight to _fallback_no_legs(), which only knows
    how to say "sitting" or "lying down" (via head-above-shoulders) --
    it has no path to "standing" or "crouching" because it never looks
    at the thighs it actually has.
    Fixed with a new middle tier, _classify_from_thighs(): when hip+
    knee are visible for at least one leg but no ankle is, it measures
    the hip->knee segment's own tilt from vertical. A vertical thigh
    (hanging straight down from the hip) means standing; a thigh
    swung out toward horizontal (forward onto a seat) means sitting --
    combined with torso lean the same way the knee-angle path already
    combines sitting vs. crouching. This is checked BEFORE
    _fallback_no_legs(), which remains the true last resort for when
    not even a hip+knee pair is visible.
  - Also loosened the "both ankles must be visible" requirement for
    the knee-angle path itself: if only one leg's full hip-knee-ankle
    chain is visible (the other ankle occluded/out of frame), that
    single leg's knee angle is now used directly instead of discarding
    the frame's leg information entirely just because the OTHER leg's
    ankle happened to be missing.

v4 (prior version):
  - FIXED false "lying down" when sitting at a desk/table with legs
    out of frame or occluded. Root cause: knee_angle() needs hip +
    knee + ankle ALL visible, which almost never happens at a desk --
    so classify_frame() was falling through to the no-legs fallback
    on nearly every desk-webcam frame, not just as a rare edge case.
    That fallback used to compare bbox_width to bbox_height directly:
    when only the upper body is visible, the detection box is just a
    CROP of whatever's in frame, and its aspect ratio reflects how
    that crop happened to be framed (arms out, elbows wide, a close
    shot) -- not the person's actual orientation. A perfectly upright
    person leaning on a desk can easily produce a box that's wider
    than tall, which used to trigger "lying down".
    Fixed by checking head-above-shoulders (see _head_above_shoulders)
    FIRST: it only needs a head keypoint (nose/eye/ear) and one
    shoulder -- both visible in nearly every desk-webcam frame -- and
    it isn't confused by cropping the way box dimensions are, since an
    upright head is above the shoulders no matter how tightly the box
    is cropped. The old bbox-aspect-ratio check is now only a last
    resort, used solely when even the head isn't visible.
  - Also guarded the PRIMARY torso-tilt "lying down" trigger (used
    when hips ARE visible) with the same head-above-shoulders signal:
    leaning in close to a laptop camera can distort the shoulder-hip
    angle past the lying-down threshold on its own (see
    torso_angle_from_vertical's existing numerical-stability caveat),
    so a clear head-above-shoulders reading now overrides a borderline
    torso-tilt reading instead of both being trusted independently.

v3 (prior version):
  - REMOVED the calibration step entirely. There is no more startup
    delay, no "Calibrating..." screen, no per-session baseline. Every
    frame is classified immediately using fixed, absolute thresholds
    (documented below). Trade-off, stated plainly: fixed thresholds are
    less precise at an extreme camera angle than a per-person baseline
    would be, but they work the instant the video starts, which is what
    was asked for.
  - REMOVED "unknown" and "legs not visible" as possible outputs. The
    classifier now always commits to one concrete label. When a signal
    is ambiguous (e.g. knee angle sits between the sitting and standing
    thresholds) it picks whichever is closer instead of punting. When
    legs aren't visible at all, it falls back to bounding-box aspect
    ratio (tall+narrow -> standing, wide+short -> lying down) so there
    is always a real answer. The only string that is NOT a posture is
    "no person detected", which is used only when the model finds
    nobody in frame at all -- that's a statement about the scene, not
    an unknown posture for a person who is there.
  - Jumping no longer needs a calibrated ground reference. An
    AdaptiveGroundEstimator keeps a rolling window of ankle height and
    uses its recent low point (feet flat on the floor) as a
    continuously self-updating "ground level" -- no explicit
    calibration phase required, it just accumulates in the background
    from frame one.

v2 (prior version, for reference):
  - Fixed a calibration hang bug (counter only advanced on detection).
  - Added jumping detection (calibrated ground reference).
  - Fixed unstable person tracking (now: largest bbox each frame).

WHAT THIS FILE DOES:
  1. Runs YOLO11-Pose on every video frame (webcam or file) -- no warm-up.
  2. Extracts 17 COCO-format keypoints for the most prominent person.
  3. Classifies posture per frame using fixed geometric thresholds:
       - jumping     (both ankles simultaneously well above their
                       recent floor-contact height)
       - running     (sustained ankle oscillation across a frame window)
       - lying down   (torso axis closer to horizontal than vertical)
       - standing    (knees close to straight, OR thighs hang vertical
                       when ankles aren't visible)
       - sitting     (knees bent, torso stays upright, OR thighs swing
                       toward horizontal when ankles aren't visible)
       - crouching   (knees bent, torso leans forward)
     Always returns exactly one of these -- never "unknown".
  4. Smooths the classification over a short rolling window so a single
     noisy frame doesn't flicker the output label.
  5. Prints/overlays the current stable posture + confidence live.

REQUIREMENTS:
    pip install ultralytics opencv-python numpy

RUN (webcam):
    python pose_estimation_final.py

RUN (video file):
    python pose_estimation_final.py --source path/to/video.mp4
"""

import argparse
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------
# COCO 17-keypoint indices (YOLO-Pose output format)
# ---------------------------------------------------------------------
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 1, 2, 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

MIN_KEYPOINT_CONF = 0.5    # keypoints below this confidence are treated as "not detected"
                            # (used for posture classification -- keep this strict, a wrong
                            # posture label is visibly wrong)
INTERACTION_MIN_KEYPOINT_CONF = 0.3   # separate, more lenient bar for wrists/elbows/hips/knees
                                        # used by the relation module. A held object partially
                                        # occludes its own wrist keypoint, which routinely drops
                                        # its confidence below 0.5 even though the model still has
                                        # a decent, usable position estimate for it. Using the
                                        # strict posture threshold here was silently dropping
                                        # exactly the keypoints (occluded-by-object wrists) that
                                        # matter most for interaction detection -- that's what was
                                        # causing held objects like phones to stop being described.
SMOOTHING_WINDOW = 10      # frames of history used for stability / motion checks
MIN_FRAMES_TO_CLASSIFY = 8  # require this many valid frames before trusting a motion-based label

# ---------------------------------------------------------------------
# Fixed thresholds (no calibration -- these are absolute, work from frame 1)
# ---------------------------------------------------------------------
STANDING_KNEE_MIN_DEG = 160   # knee angle at/above this -> leg is basically straight
SITTING_KNEE_MAX_DEG = 115    # knee angle at/below this -> leg is clearly bent (sitting/crouching range)
CROUCH_TORSO_MIN_DEG = 30     # torso tilt at/above this, with bent knees -> leaning forward = crouching
LYING_DOWN_ABS_DEG = 55       # torso axis this far from vertical is horizontal, regardless of camera setup
JUMP_ABOVE_GROUND_RATIO = 0.12  # both ankles must rise at least this fraction of bbox height
                                  # above their recent floor-contact height, simultaneously, to count as airborne
GROUND_WINDOW_FRAMES = 90     # ~3s at 30fps: how far back we look for "feet were flat on the floor"
MIN_TORSO_SEGMENT_FRAC = 0.15  # the shoulder-mid -> hip-mid vector must be at least this fraction of
                                 # bbox height, or its ANGLE is numerically unstable (see comment below)
                                 # and gets discarded rather than trusted
HEAD_ABOVE_SHOULDER_MARGIN_SHOULDER_RATIO = 0.25  # how far above the shoulders (as a fraction of
                                 # shoulder-to-shoulder width, a stable camera-distance-relative unit)
                                 # the head needs to be before we trust "upright" over pixel jitter
HEAD_ABOVE_SHOULDER_MARGIN_BBOX_RATIO = 0.10  # same idea, fraction of bbox height instead, used only
                                 # when just one shoulder is visible (no shoulder-width unit available)
MIN_THIGH_SEGMENT_FRAC = 0.12  # the hip -> knee vector must be at least this fraction of bbox height,
                                 # or its angle is treated as noise rather than signal (same rationale
                                 # as MIN_TORSO_SEGMENT_FRAC, just for the shorter thigh segment)
THIGH_STANDING_MAX_DEG = 25    # thigh tilt from vertical at/below this -> thigh hangs essentially
                                 # straight down from the hip, consistent with standing
THIGH_SITTING_MIN_DEG = 50     # thigh tilt from vertical at/above this -> thigh is swung out toward
                                 # horizontal (forward onto a seat), consistent with sitting


@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass
class PersonPose:
    keypoints: list          # length-17 list of Optional[Keypoint]
    bbox_height: float       # person's detection box height in pixels
    bbox_width: float        # person's detection box width in pixels


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------

def angle_deg(a, b, c) -> float:
    """Angle at point b formed by rays b->a and b->c, in degrees."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)
    if mag_ba == 0 or mag_bc == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def get_kp(pose: PersonPose, idx: int) -> Optional[Keypoint]:
    kp = pose.keypoints[idx]
    if kp is None or kp.confidence < MIN_KEYPOINT_CONF:
        return None
    return kp


def knee_angle(pose: PersonPose, side: str) -> Optional[float]:
    hip = get_kp(pose, LEFT_HIP if side == "left" else RIGHT_HIP)
    knee = get_kp(pose, LEFT_KNEE if side == "left" else RIGHT_KNEE)
    ankle = get_kp(pose, LEFT_ANKLE if side == "left" else RIGHT_ANKLE)
    if not all([hip, knee, ankle]):
        return None
    return angle_deg((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))


def thigh_angle_from_vertical(pose: PersonPose, side: str) -> Optional[float]:
    """Angle of the hip->knee segment from vertical, for one leg. ~0 deg =
    thigh hangs straight down (standing), ~90 deg = thigh horizontal
    (typically sitting, thigh swung forward onto a seat).

    This exists for frames where the ankle isn't visible (cropped out
    of a webcam frame, or hidden under a desk) so knee_angle() can't be
    computed at all -- but the thigh itself (hip+knee) still is. The
    thigh's own tilt is a real, independent standing-vs-sitting signal
    on its own; it doesn't need the shin/ankle to confirm it.

    Same numerical-stability guard as torso_angle_from_vertical(): a
    short hip->knee vector (close-range camera, tightly cropped frame)
    means a couple of pixels of ordinary keypoint jitter can swing the
    angle wildly, so a too-short segment returns None instead of a
    noisy number the caller might otherwise trust.
    """
    hip = get_kp(pose, LEFT_HIP if side == "left" else RIGHT_HIP)
    knee = get_kp(pose, LEFT_KNEE if side == "left" else RIGHT_KNEE)
    if hip is None or knee is None:
        return None
    dx = knee.x - hip.x
    dy = knee.y - hip.y
    segment_len = math.hypot(dx, dy)
    if segment_len < MIN_THIGH_SEGMENT_FRAC * pose.bbox_height:
        return None  # too short to trust -- angle would be noise, not signal
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def torso_angle_from_vertical(pose: PersonPose) -> Optional[float]:
    """Angle of the shoulder-hip axis from vertical. ~0 deg = upright, ~90 deg = horizontal (lying down).

    IMPORTANT numerical-stability guard: this angle comes from a single
    2D vector (shoulder-midpoint -> hip-midpoint). When that vector is
    SHORT -- e.g. a close-range laptop webcam where the visible torso
    between shoulders and hips only spans a few pixels -- a couple of
    pixels of ordinary keypoint jitter swings the angle wildly. A tiny,
    almost-vertical vector with a few noisy pixels of horizontal wobble
    can easily read as 60-80 degrees, which is exactly what was causing
    constant false "lying down" calls on normal sitting/standing webcam
    footage. So: if the vector is too short relative to the person's own
    bbox height to be trustworthy, we return None instead of a noisy
    angle -- the caller then falls through to other checks instead of
    acting on a number that isn't meaningful.
    """
    l_sh, r_sh = get_kp(pose, LEFT_SHOULDER), get_kp(pose, RIGHT_SHOULDER)
    l_hip, r_hip = get_kp(pose, LEFT_HIP), get_kp(pose, RIGHT_HIP)
    if not all([l_sh, r_sh, l_hip, r_hip]):
        return None
    shoulder_mid = ((l_sh.x + r_sh.x) / 2, (l_sh.y + r_sh.y) / 2)
    hip_mid = ((l_hip.x + r_hip.x) / 2, (l_hip.y + r_hip.y) / 2)
    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    segment_len = math.hypot(dx, dy)
    if segment_len < MIN_TORSO_SEGMENT_FRAC * pose.bbox_height:
        return None  # too short to trust -- angle would be noise, not signal
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def _head_above_shoulders(pose: PersonPose) -> Optional[bool]:
    """
    Whether the head sits meaningfully ABOVE the shoulders in the image
    (smaller y = higher up in image coordinates) -- a cheap, robust
    proxy for "torso is upright" that only needs a head keypoint
    (nose, or an eye/ear if the nose isn't visible) plus at least one
    shoulder. Both survive far more real-world occlusion and cropping
    than the hip+knee+ankle chain knee_angle() needs, which is exactly
    why this exists: sitting at a desk with legs hidden under/behind
    the table (or simply out of a laptop webcam's frame) is the NORMAL
    case for a desk session, not a rare edge case, so this signal
    needs to hold up under partial visibility, not just full-body shots.

    Unlike comparing bbox width to bbox height, this isn't fooled by
    cropping: an upright person's head is above their shoulders no
    matter how tightly the detection box happens to be cropped (arms
    out on a desk, a close framing, whatever). Box aspect ratio, by
    contrast, reflects how the visible portion happened to be framed,
    not the person's actual orientation -- that mismatch is what was
    causing "sitting at a table" to read as "lying down".

    Returns:
        True  -- head clearly above shoulders (upright)
        False -- head roughly level with or below the shoulders
                  (consistent with lying on your side facing the
                  camera, where head and shoulders sit at a similar
                  image height)
        None  -- not enough visible (no head keypoint, or no shoulder
                  keypoint at all) to say either way
    """
    head = (get_kp(pose, NOSE) or get_kp(pose, LEFT_EYE) or get_kp(pose, RIGHT_EYE)
            or get_kp(pose, LEFT_EAR) or get_kp(pose, RIGHT_EAR))
    l_sh, r_sh = get_kp(pose, LEFT_SHOULDER), get_kp(pose, RIGHT_SHOULDER)
    if head is None or (l_sh is None and r_sh is None):
        return None

    shoulder_ys = [s.y for s in (l_sh, r_sh) if s is not None]
    shoulder_y = sum(shoulder_ys) / len(shoulder_ys)

    if l_sh is not None and r_sh is not None:
        # Both shoulders visible: use shoulder-to-shoulder width as the
        # unit for the margin -- it scales naturally with how close the
        # person is to the camera, the same way bbox_height is used
        # elsewhere in this file.
        shoulder_span = math.hypot(l_sh.x - r_sh.x, l_sh.y - r_sh.y)
        margin = HEAD_ABOVE_SHOULDER_MARGIN_SHOULDER_RATIO * shoulder_span
    else:
        # Only one shoulder visible -- no shoulder-width unit available,
        # fall back to a fraction of bbox height instead.
        margin = HEAD_ABOVE_SHOULDER_MARGIN_BBOX_RATIO * pose.bbox_height

    return (shoulder_y - head.y) > margin


# ---------------------------------------------------------------------
# Adaptive ground-level estimate (replaces the old calibration step)
# ---------------------------------------------------------------------
# No explicit "stand still for 3 seconds" phase. Instead, this keeps a
# rolling window of ankle height and treats the lowest point reached
# recently (largest pixel-y = feet planted on the floor) as "ground".
# It self-updates continuously, so it needs no startup wait and adapts
# if the person or camera moves.

class AdaptiveGroundEstimator:
    def __init__(self, window_frames: int = GROUND_WINDOW_FRAMES):
        self.samples: deque = deque(maxlen=window_frames)

    def update(self, pose: PersonPose):
        l = get_kp(pose, LEFT_ANKLE)
        r = get_kp(pose, RIGHT_ANKLE)
        if l is not None and r is not None:
            self.samples.append((l.y + r.y) / 2)

    def ground_y(self) -> Optional[float]:
        if not self.samples:
            return None
        return max(self.samples)  # largest y = lowest point on screen = feet on the floor


# ---------------------------------------------------------------------
# Posture classifier
# ---------------------------------------------------------------------

class PostureClassifier:
    """
    Classifies posture every frame using fixed geometric thresholds plus
    a short rolling history for motion-dependent checks (running,
    jumping) and for label smoothing. Always returns a concrete label --
    never "unknown" -- by falling back through progressively less leg
    information:
      1. Full knee angle (hip+knee+ankle) on at least one leg -- the
         most precise signal.
      2. Thigh tilt from vertical (hip+knee only, no ankle needed) --
         see _classify_from_thighs / thigh_angle_from_vertical. Covers
         the common case of ankles being cropped out or under a desk
         while the thighs are still visible.
      3. Head-above-shoulders / bbox aspect ratio (see
         _fallback_no_legs) -- true last resort, used only when not
         even a hip+knee pair is visible.
    """

    LABELS = ("standing", "sitting", "crouching", "lying down", "running", "jumping")

    def __init__(self, smoothing_window: int = SMOOTHING_WINDOW):
        self.history: deque = deque(maxlen=smoothing_window)
        self.label_history: deque = deque(maxlen=smoothing_window)
        self.ground = AdaptiveGroundEstimator()

    def _is_jumping(self, pose: PersonPose) -> bool:
        ground_y = self.ground.ground_y()
        if ground_y is None:
            return False
        l = get_kp(pose, LEFT_ANKLE)
        r = get_kp(pose, RIGHT_ANKLE)
        if l is None or r is None:
            return False
        threshold = JUMP_ABOVE_GROUND_RATIO * pose.bbox_height
        return (ground_y - l.y) > threshold and (ground_y - r.y) > threshold

    def _is_running(self) -> bool:
        """Sustained ankle vertical oscillation across the window.
        Requires every frame in the window to have CONFIDENT ankle
        keypoints -- low-confidence/occluded ankles jitter and cause
        false positives, e.g. legs out of webcam frame."""
        if len(self.history) < MIN_FRAMES_TO_CLASSIFY:
            return False

        l_ys, r_ys = [], []
        for pose in self.history:
            l = get_kp(pose, LEFT_ANKLE)
            r = get_kp(pose, RIGHT_ANKLE)
            if l is None or r is None:
                return False
            l_ys.append(l.y)
            r_ys.append(r.y)

        bbox_height = self.history[-1].bbox_height
        threshold = 0.12 * bbox_height
        return (max(l_ys) - min(l_ys)) > threshold or (max(r_ys) - min(r_ys)) > threshold

    @staticmethod
    def _classify_from_thighs(pose: PersonPose, torso_tilt: Optional[float]) -> Optional[str]:
        """
        Used when NEITHER leg has a full hip-knee-ankle chain visible
        (so knee_angle() is None for both sides) but at least one thigh
        (hip+knee) still is -- e.g. shins/ankles cropped out of a
        webcam frame or hidden under a desk while the thighs themselves
        are in view. This is the common "sitting or standing close to
        a desk camera" case, and it's a real signal on its own: a
        thigh hanging essentially straight down from the hip means
        standing, while a thigh swung out toward horizontal (forward
        onto a seat) means sitting -- neither needs the shin/ankle to
        confirm it.

        Combines with torso lean the same way the knee-angle path
        already does: a vertical thigh with a forward-leaning torso
        reads as crouching rather than standing, and likewise for
        sitting vs. crouching.

        Returns None (not a guess) when the averaged thigh angle falls
        between the standing and sitting thresholds -- genuinely
        ambiguous with this little information -- so the caller falls
        through to the head/shoulders last resort instead of forcing a
        possibly-wrong standing/sitting call from a borderline angle.
        """
        angles = []
        for side in ("left", "right"):
            a = thigh_angle_from_vertical(pose, side)
            if a is not None:
                angles.append(a)
        if not angles:
            return None
        avg_thigh = sum(angles) / len(angles)

        if avg_thigh <= THIGH_STANDING_MAX_DEG:
            if torso_tilt is not None and torso_tilt >= CROUCH_TORSO_MIN_DEG:
                return "crouching"
            return "standing"

        if avg_thigh >= THIGH_SITTING_MIN_DEG:
            if torso_tilt is not None and torso_tilt >= CROUCH_TORSO_MIN_DEG:
                return "crouching"
            return "sitting"

        return None  # between thresholds -- not enough to commit, let caller fall back further

    @staticmethod
    def _fallback_no_legs(pose: PersonPose) -> str:
        """
        Used when NOTHING leg-based worked -- no full knee angle on
        either side AND no usable thigh (hip+knee) on either side
        either. In practice this means only the upper body (or even
        less) is visible: e.g. a very tight face/shoulders crop.

        Head-above-shoulders (see _head_above_shoulders) is checked
        FIRST, since it only needs a head keypoint and one shoulder --
        both visible in nearly every desk-webcam frame -- and isn't
        thrown off by how tightly the box happens to be cropped, unlike
        comparing bbox width to bbox height directly (the old approach,
        which was misreading "sitting upright at a table with arms out
        of frame" as "lying down" purely because that crop's box came
        out wider than tall).

        The bbox-aspect-ratio check is now only a last resort, used
        solely when even the head isn't visible -- e.g. only a hand or
        forearm is in frame. In that situation this still defaults to
        "sitting" unless the box is unambiguously wide (at least 30%
        wider than tall), since sitting is overwhelmingly the more
        common real case when this little is visible.
        """
        upright = _head_above_shoulders(pose)
        if upright is True:
            return "sitting"
        if upright is False:
            return "lying down"

        # Last resort: neither a head keypoint nor any shoulder was
        # usable at all -- fall back to the old, cruder box-shape check.
        if pose.bbox_width > 1.3 * pose.bbox_height:
            return "lying down"
        return "sitting"

    def classify_frame(self, pose: PersonPose) -> str:
        """Raw per-frame classification. Always returns a real label."""
        self.history.append(pose)
        self.ground.update(pose)

        # Jumping checked first: it's the most specific / short-lived event,
        # and both-feet-off-ground is a stronger signal than generic ankle
        # oscillation (which running also produces).
        if self._is_jumping(pose):
            return "jumping"

        if self._is_running():
            return "running"

        torso_tilt = torso_angle_from_vertical(pose)
        head_upright = _head_above_shoulders(pose)
        # A clear head-above-shoulders reading overrides a borderline
        # torso-tilt reading rather than trusting torso_tilt alone --
        # leaning in close to a laptop camera can distort the shoulder-
        # hip angle past this threshold on its own (see
        # torso_angle_from_vertical's numerical-stability note above),
        # and the head/shoulder relationship is the more direct, less
        # noise-prone of the two signals when both are available.
        if (torso_tilt is not None and torso_tilt > LYING_DOWN_ABS_DEG
                and head_upright is not True):
            return "lying down"

        l_knee = knee_angle(pose, "left")
        r_knee = knee_angle(pose, "right")

        # Use whichever leg(s) actually have a full hip-knee-ankle chain
        # visible -- average both if both are there, but don't discard a
        # perfectly good single-leg reading just because the OTHER leg's
        # ankle happens to be occluded or out of frame.
        if l_knee is not None and r_knee is not None:
            avg_knee = (l_knee + r_knee) / 2
        elif l_knee is not None:
            avg_knee = l_knee
        elif r_knee is not None:
            avg_knee = r_knee
        else:
            avg_knee = None

        if avg_knee is None:
            # No full knee angle on either leg -- try the thigh-only
            # signal (hip+knee, no ankle needed) before giving up on leg
            # information entirely.
            thigh_label = self._classify_from_thighs(pose, torso_tilt)
            if thigh_label is not None:
                return thigh_label
            return self._fallback_no_legs(pose)

        if avg_knee >= STANDING_KNEE_MIN_DEG:
            return "standing"

        if avg_knee <= SITTING_KNEE_MAX_DEG:
            # legs are clearly bent -- upright torso means sitting,
            # forward-leaning torso means crouching. If torso isn't
            # visible either, default to sitting (more common than
            # crouching when only the lower body is in frame).
            if torso_tilt is not None and torso_tilt >= CROUCH_TORSO_MIN_DEG:
                return "crouching"
            return "sitting"

        # Between the two thresholds: genuinely in-between, but we never
        # say "unknown" -- pick whichever side it's numerically closer to.
        dist_to_standing = abs(avg_knee - STANDING_KNEE_MIN_DEG)
        dist_to_sitting = abs(avg_knee - SITTING_KNEE_MAX_DEG)
        if dist_to_standing <= dist_to_sitting:
            return "standing"
        if torso_tilt is not None and torso_tilt >= CROUCH_TORSO_MIN_DEG:
            return "crouching"
        return "sitting"

    def classify_smoothed(self, pose: PersonPose) -> tuple:
        """
        Returns (label, confidence) where confidence is the fraction of
        recent frames that agreed with the returned label -- a simple,
        honest measure of how stable the current classification is,
        not a model-derived probability.
        """
        raw_label = self.classify_frame(pose)
        self.label_history.append(raw_label)

        counts = {}
        for lbl in self.label_history:
            counts[lbl] = counts.get(lbl, 0) + 1
        best_label = max(counts, key=counts.get)
        stability = counts[best_label] / len(self.label_history)

        return best_label, round(stability * 100, 1)


# ---------------------------------------------------------------------
# YOLO11-Pose result -> PersonPose conversion
# ---------------------------------------------------------------------

def get_interaction_keypoints(pose: PersonPose) -> list:
    """
    Returns (x, y, kind) tuples for the keypoints most relevant to
    human-object interaction -- wrists and elbows for anything held or
    used, hips and knees for anything sat/leaned on -- skipping any
    that weren't detected above INTERACTION_MIN_KEYPOINT_CONF (a lower
    bar than posture classification uses, since held objects routinely
    depress their own wrist keypoint's confidence).

    `kind` is one of "wrist", "elbow", "hip", "knee" -- kept alongside
    the coordinates (not just returning bare points) because it's a
    useful feature for the interaction classifier: "object near a
    wrist" and "object near a knee" are very different evidence for
    whether real contact is happening, even at the same pixel
    distance.
    """
    relevant = [
        (LEFT_WRIST, "wrist"), (RIGHT_WRIST, "wrist"),
        (LEFT_ELBOW, "elbow"), (RIGHT_ELBOW, "elbow"),
        (LEFT_HIP, "hip"), (RIGHT_HIP, "hip"),
        (LEFT_KNEE, "knee"), (RIGHT_KNEE, "knee"),
    ]
    points = []
    for idx, kind in relevant:
        kp = pose.keypoints[idx]
        if kp is not None and kp.confidence >= INTERACTION_MIN_KEYPOINT_CONF:
            points.append((kp.x, kp.y, kind))
    return points


def get_face_point(pose: PersonPose) -> Optional[tuple]:
    """
    Returns (x, y) for the best-available face landmark -- nose first,
    then an eye, then an ear -- or None if no face keypoint was
    detected above INTERACTION_MIN_KEYPOINT_CONF this frame.

    This exists purely to give the HOI relation module visual context
    of the face region (e.g. so a crop can include the mouth/ear area
    for things like "drinking" or "on the phone"). It is NOT used by
    posture classification and is NOT part of the interaction
    classifier's feature vector.
    """
    for idx in (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR):
        kp = pose.keypoints[idx]
        if kp is not None and kp.confidence >= INTERACTION_MIN_KEYPOINT_CONF:
            return (kp.x, kp.y)
    return None


def pose_result_to_bbox(pose_result) -> Optional[list]:
    """Returns [x1, y1, x2, y2] for the SAME person pose_result_to_person
    tracks (largest bbox each frame) -- so the relation classifier pairs
    against the exact person the posture classifier is already using,
    instead of asking a second detector to re-find 'person' independently
    (which can miss at a different confidence than the pose model)."""
    if pose_result.boxes is None or len(pose_result.boxes) == 0:
        return None
    boxes = pose_result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = int(areas.argmax())
    return boxes[idx].tolist()


def pose_result_to_person(pose_result) -> Optional[PersonPose]:
    """Picks the largest (most prominent / closest) detected person each
    frame. Detection index order is NOT guaranteed stable across frames
    when multiple people are present, so always reading index 0 could
    silently swap who's being tracked -- largest bbox is a much steadier
    proxy for 'the subject' in a single-subject setup."""
    if pose_result.keypoints is None or pose_result.boxes is None or len(pose_result.boxes) == 0:
        return None

    boxes = pose_result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = int(areas.argmax())

    xy = pose_result.keypoints.xy[idx].cpu().numpy()
    if pose_result.keypoints.conf is not None:
        conf = pose_result.keypoints.conf[idx].cpu().numpy()
    else:
        conf = np.ones(len(xy))  # some export paths omit per-keypoint confidence

    keypoints = []
    for i in range(17):
        x, y = xy[i]
        c = float(conf[i])
        keypoints.append(None if (x == 0 and y == 0) else Keypoint(float(x), float(y), c))

    bbox_height = float(boxes[idx][3] - boxes[idx][1])
    bbox_width = float(boxes[idx][2] - boxes[idx][0])
    return PersonPose(keypoints=keypoints, bbox_height=bbox_height, bbox_width=bbox_width)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="YOLO11-Pose posture classification")
    parser.add_argument("--source", type=str, default="0",
                         help="Camera index (default '0') or path to a video file")
    parser.add_argument("--debug", action="store_true",
                         help="Print per-keypoint confidence every 15 frames, to diagnose a classification")
    args = parser.parse_args()

    source = int(args.source) if args.source.lstrip("-").isdigit() else args.source

    print("Loading YOLO11-Pose model...")
    pose_model = YOLO("yolo11n-pose.pt")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    classifier = PostureClassifier()
    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1

            results = pose_model(frame, verbose=False)
            person = pose_result_to_person(results[0])
            annotated = results[0].plot()

            if person is None:
                label, stability = "no person detected", 0.0
            else:
                label, stability = classifier.classify_smoothed(person)

            if args.debug and frame_count % 15 == 0:
                print(f"\n--- frame {frame_count} ---")
                if person is None:
                    print("NO PERSON DETECTED AT ALL (pose model found nobody in frame)")
                else:
                    names = {LEFT_SHOULDER: "L_shoulder", RIGHT_SHOULDER: "R_shoulder",
                              LEFT_HIP: "L_hip", RIGHT_HIP: "R_hip",
                              LEFT_KNEE: "L_knee", RIGHT_KNEE: "R_knee",
                              LEFT_ANKLE: "L_ankle", RIGHT_ANKLE: "R_ankle"}
                    for idx, name in names.items():
                        kp = person.keypoints[idx]
                        if kp is None:
                            print(f"  {name}: NOT DETECTED")
                        else:
                            flag = "OK" if kp.confidence >= MIN_KEYPOINT_CONF else "TOO LOW CONF"
                            print(f"  {name}: conf={kp.confidence:.2f} ({flag})")
                    lk = knee_angle(person, "left")
                    rk = knee_angle(person, "right")
                    tl = thigh_angle_from_vertical(person, "left")
                    tr = thigh_angle_from_vertical(person, "right")
                    tt = torso_angle_from_vertical(person)
                    hu = _head_above_shoulders(person)
                    print(f"  computed left_knee_angle: {lk}")
                    print(f"  computed right_knee_angle: {rk}")
                    print(f"  computed left_thigh_angle: {tl}")
                    print(f"  computed right_thigh_angle: {tr}")
                    print(f"  computed torso_tilt: {tt}")
                    print(f"  computed head_above_shoulders: {hu}")
                    print(f"  -> raw label this frame: {classifier.label_history[-1] if classifier.label_history else 'n/a'}")

            text = f"{label} ({stability}% stable)"
            cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

            cv2.imshow("Pose Estimation - Posture Classification (press q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
