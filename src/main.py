import time
import cv2
from ultralytics import YOLO

from object_detection import detect_objects, split_persons_and_objects
from interaction_engine import InteractionEngine

from pose_estimation import (
    pose_result_to_person,
    get_interaction_keypoints,
    get_face_point,
    PostureClassifier,
    knee_angle,
    thigh_angle_from_vertical,
    torso_angle_from_vertical,
)


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

SOURCE = 0
INFERENCE_IMGSZ = 640

OBJECT_DETECTION_EVERY_N_FRAMES = 2
DEBUG_EVERY_N_FRAMES = 30


# ------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------

print("Loading YOLO11-Pose model...")
pose_model = YOLO("models/yolo11n-pose.pt")

print("Loading object detection model...")

classifier = PostureClassifier()
interaction_engine = InteractionEngine()


# ------------------------------------------------------------
# CAMERA
# ------------------------------------------------------------

cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# ------------------------------------------------------------
# CACHED OBJECT DETECTIONS
# ------------------------------------------------------------

frame_id = 0
last_detections = []
last_objects = []


# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------

try:
    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_id += 1

        # ----------------------------------------------------
        # 1. POSE ESTIMATION
        # ----------------------------------------------------

        pose_results = pose_model(
            frame,
            verbose=False,
            imgsz=INFERENCE_IMGSZ
        )

        person = pose_result_to_person(pose_results[0])

        # ----------------------------------------------------
        # 2. POSTURE CLASSIFICATION
        # ----------------------------------------------------

        if person is not None:
            posture, stability = classifier.classify_smoothed(person)
        else:
            posture = "no person detected"
            stability = 0.0

        # ----------------------------------------------------
        # 3. OBJECT DETECTION
        # ----------------------------------------------------

        if frame_id % OBJECT_DETECTION_EVERY_N_FRAMES == 0:

            last_detections = detect_objects(
                frame,
                conf_threshold=0.20,
                imgsz=INFERENCE_IMGSZ
            )

            _, last_objects = split_persons_and_objects(
                last_detections
            )

        objects = last_objects

        # ----------------------------------------------------
        # 4. HUMAN-OBJECT INTERACTION
        # ----------------------------------------------------

        interaction_events = []

        if person is not None:

            wrist_points = get_interaction_keypoints(person)
            face_point = get_face_point(person)

            interaction_events = interaction_engine.update(
                detections=objects,
                wrist_points=wrist_points,
                face_point=face_point,
                bbox_height=max(person.bbox_height, 1.0),
                now=time.time(),
            )

        # ----------------------------------------------------
        # 5. DEBUG INFORMATION
        # ----------------------------------------------------

        if frame_id % DEBUG_EVERY_N_FRAMES == 0:

            print(
                f"\n[frame {frame_id}] "
                f"posture={posture} "
                f"stability={stability}%"
            )

            print(
                f"objects={[o['object'] for o in objects]}"
            )

            print(
                f"interactions={[event.action for event in interaction_events]}"
            )

            if person is not None:
                print(
                    f"knees: "
                    f"L={knee_angle(person, 'left')} "
                    f"R={knee_angle(person, 'right')}"
                )

                print(
                    f"thighs: "
                    f"L={thigh_angle_from_vertical(person, 'left')} "
                    f"R={thigh_angle_from_vertical(person, 'right')}"
                )

                print(
                    f"torso={torso_angle_from_vertical(person)}"
                )

        # ----------------------------------------------------
        # 6. DRAW POSE
        # ----------------------------------------------------

        annotated_frame = pose_results[0].plot()

        # ----------------------------------------------------
        # 7. DRAW OBJECT DETECTIONS
        # ----------------------------------------------------

        for detection in last_detections:

            object_name = detection["object"]
            confidence = detection["confidence"]

            x1, y1, x2, y2 = [
                int(v) for v in detection["bbox"]
            ]

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            label = f"{object_name} {confidence:.2f}"

            cv2.putText(
                annotated_frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        # ----------------------------------------------------
        # 8. DRAW POSTURE
        # ----------------------------------------------------

        posture_text = (
            f"Posture: {posture} "
            f"({stability}% stable)"
        )

        cv2.putText(
            annotated_frame,
            posture_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        # ----------------------------------------------------
        # 9. DRAW INTERACTION RESULTS
        # ----------------------------------------------------

        y_position = 60

        for event in interaction_events:

            cv2.putText(
                annotated_frame,
                f"Action: {event.action}",
                (10, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 200, 0),
                2
            )

            y_position += 25

            cv2.putText(
                annotated_frame,
                f"Confidence: {event.confidence:.0f}%",
                (10, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 0),
                2
            )

            y_position += 25

            for reason in event.reasons:

                cv2.putText(
                    annotated_frame,
                    reason,
                    (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 200, 0),
                    1
                )

                y_position += 20

            y_position += 5

        # ----------------------------------------------------
        # 10. DISPLAY
        # ----------------------------------------------------

        cv2.imshow(
            "Project V2C - Human Action Recognition",
            annotated_frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break


finally:

    cap.release()
    cv2.destroyAllWindows()

    print("\nCamera stopped.")
