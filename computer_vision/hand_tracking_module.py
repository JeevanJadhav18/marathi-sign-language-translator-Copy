
import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path


class HandTracker:

    def __init__(self):

        model_path = (
            Path(__file__).resolve().parent
            / "hand_landmarker.task"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                "hand_landmarker.task not found"
            )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2
        )

        self.detector = (
            mp.tasks.vision.HandLandmarker
            .create_from_options(options)
        )

        self.last_timestamp = -1

        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

    def process(self, frame):

        # Mirror the webcam frame
        frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]

        rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb)
        )

        timestamp = int(time.monotonic() * 1000)
        timestamp = max(
            timestamp, self.last_timestamp + 1
        )
        self.last_timestamp = timestamp

        result = self.detector.detect_for_video(
            image, timestamp
        )

        features = np.zeros(
            (2, 21, 3), dtype=np.float32
        )

        presence = np.zeros(2, dtype=np.int8)

        for i, hand in enumerate(result.hand_landmarks):

            label = result.handedness[i][0].category_name

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

            for start, end in self.connections:

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

        feature_vector = features.flatten()

        return frame, feature_vector, presence

    def close(self):
        self.detector.close()
