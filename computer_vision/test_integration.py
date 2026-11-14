
import cv2
import numpy as np
from hand_tracking_module import HandTracker

tracker = HandTracker()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    tracker.close()
    raise RuntimeError("Camera not found")

try:
    while True:

        success, frame = cap.read()

        if not success:
            break

        # Get data from Computer Vision module
        frame, features, presence = tracker.process(frame)

        cv2.putText(
            frame,
            f"Features: {len(features)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Integration Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            np.savez(
                "integration_output.npz",
                features=features,
                presence=presence
            )

            print("Features Shape:", features.shape)
            print("Hands Present:", presence)

        elif key == ord("q"):
            break

finally:
    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
