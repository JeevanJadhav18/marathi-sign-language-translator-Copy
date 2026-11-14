
import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path

# Model path
folder = Path(__file__).resolve().parent
model_path = folder / "hand_landmarker.task"

# Configure MediaPipe
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(model_path)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2
)

# Hand landmark connections
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam not found")

previous_time = time.monotonic()
last_timestamp = -1

with mp.tasks.vision.HandLandmarker.create_from_options(
    options
) as detector:

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb)
        )

        timestamp = int(time.monotonic() * 1000)
        timestamp = max(timestamp, last_timestamp + 1)
        last_timestamp = timestamp

        # Detect hands
        result = detector.detect_for_video(
            mp_image, timestamp
        )

        # Fixed data for left and right hands
        features = np.zeros(
            (2, 21, 3), dtype=np.float32
        )
        presence = np.zeros(2, dtype=np.int8)

        for i, hand in enumerate(result.hand_landmarks):

            label = result.handedness[i][0].category_name

            # Left = 0, Right = 1
            slot = 0 if label == "Left" else 1

            coordinates = np.array(
                [[p.x, p.y, p.z] for p in hand],
                dtype=np.float32
            )

            features[slot] = coordinates
            presence[slot] = 1

            points = []

            for landmark in hand:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))

                cv2.circle(
                    frame, (x, y), 5,
                    (0, 255, 0), -1
                )

            for start, end in connections:
                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 0, 0),
                    2
                )

            cv2.putText(
                frame,
                label,
                points[0],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        # Convert to 126 features
        feature_vector = features.flatten()

        # Calculate FPS
        current_time = time.monotonic()
        elapsed = current_time - previous_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        previous_time = current_time

        # Display tracking information
        cv2.putText(
            frame,
            f"Hands: {len(result.hand_landmarks)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2
        )

        cv2.putText(
            frame,
            f"Features: {len(feature_vector)}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2
        )

        cv2.imshow("Real-Time Hand Tracking", frame)

        key = cv2.waitKey(1) & 0xFF

        # Save current frame features
        if key == ord("s"):

            np.savez(
                folder / "realtime_features.npz",
                features=feature_vector,
                presence=presence
            )

            print("Features saved!")
            print("Total Features:", len(feature_vector))
            print("Hands Present:", presence)

        # Exit webcam
        elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
