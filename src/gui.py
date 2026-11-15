from translation import translate_to_marathi
from speech import speak_marathi

import tkinter as tk

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from computer_vision.hand_tracking_module import HandTracker

import cv2
from PIL import Image, ImageTk

# 1. Create the main application window
root = tk.Tk()
tracker = None

cap = None
camera_running = False
camera_image = None

root.title("Marathi Sign Language Translator")
root.geometry("900x850")


# 2. Display application heading
heading = tk.Label(
    root,
    text="Marathi Sign Language Translator",
    font=("Arial", 20, "bold")
)

heading.pack(pady=30)


# 3. Create camera preview section
camera_frame = tk.Frame(
    root,
    width=600,
    height=350,
    bg="lightblue",
    relief="solid",
    bd=2
)

camera_frame.pack(pady=20)
camera_frame.pack_propagate(False)


camera_label = tk.Label(
    camera_frame,
    text="Camera Preview",
    font=("Arial", 18, "bold"),
    bg="lightblue"
)

camera_label.pack(expand=True)

status_label = tk.Label(
    root,
    text="Status: No Hand Detected",
    font=("Arial", 14, "bold"),
    fg="red"
)

status_label.pack(pady=5)

# 4. Create Marathi translation section
translation_frame = tk.Frame(
    root,
    width=600,
    height=100,
    bg="white",
    relief="solid",
    bd=2
)

translation_frame.pack(pady=15)
translation_frame.pack_propagate(False)


translation_label = tk.Label(
    translation_frame,
    text="Translated Text: नमस्कार",
    font=("Arial", 18, "bold"),
    bg="white"
)

translation_label.pack(expand=True)

def update_camera():
    global camera_image

    if camera_running and cap is not None:
        success, frame = cap.read()

        if success:
            if tracker is not None:
                frame, features, presence = tracker.process(frame)
                if presence.any():
                        status_label.config(
                            text="Status: Hand Detected",
                            fg="green"
                        )
                else:
                        status_label.config(
                            text="Status: No Hand Detected",
                            fg="red"
                        )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = Image.fromarray(frame)
            image = image.resize((600, 350))

            camera_image = ImageTk.PhotoImage(image)

            camera_label.config(
                image=camera_image,
                text=""
            )

        root.after(10, update_camera)

# 5. Define Start Camera function
def start_camera():
    global tracker, cap, camera_running

    if camera_running:
        return

    tracker = HandTracker()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Unable to open camera")
        tracker.close()
        tracker = None
        cap = None
        return

    camera_running = True

    print("Camera Started")

    update_camera()


# 6. Define Stop Camera function
def stop_camera():
    global tracker, cap, camera_running

    camera_running = False

    if cap is not None:
        cap.release()
        cap = None

    if tracker is not None:
        tracker.close()
        tracker = None

    camera_label.config(
        image="",
        text="Camera Preview"
    )

    print("Camera Stopped")


# 7. Create button section
button_frame = tk.Frame(root)
button_frame.pack(pady=15)


# 8. Create Start Camera button
start_button = tk.Button(
    button_frame,
    text="Start Camera",
    font=("Arial", 12, "bold"),
    width=15,
    command=start_camera
)

start_button.pack(side="left", padx=10)


# 9. Create Stop Camera button
stop_button = tk.Button(
    button_frame,
    text="Stop Camera",
    font=("Arial", 12, "bold"),
    width=15,
    command=stop_camera
)

stop_button.pack(side="left", padx=10)
speak_button = tk.Button(
    button_frame,
    text="Speak Marathi",
    font=("Arial", 12, "bold"),
    width=15,
    command=lambda: speak_marathi(translation_label.cget("text").replace("Translated Text: ", ""))
)

speak_button.pack(side="left", padx=10)


# 10. Run the application
root.mainloop()
