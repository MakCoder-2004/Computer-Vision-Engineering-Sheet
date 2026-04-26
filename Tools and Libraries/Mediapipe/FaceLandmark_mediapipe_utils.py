import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision


# ==========================
# Load Model
# ==========================
def load_detection_model(model_path):

    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=2
    )

    return vision.FaceLandmarker.create_from_options(options)


# ==========================
# Detect Frame
# ==========================
def detect_frame(landmarker, frame, timestamp_ms):

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    return landmarker.detect_for_video(mp_image, timestamp_ms)


# ==========================
# Visualize
# ==========================
def visualize_detection(frame, detection_result):

    if not detection_result.face_landmarks:
        return frame

    height, width = frame.shape[:2]

    for face_landmarks in detection_result.face_landmarks:

        points = np.array(
            [(lm.x * width, lm.y * height) for lm in face_landmarks],
            dtype=np.int32
        )

        for point in points:
            cv2.circle(frame, tuple(point), 1, (0, 255, 0), -1)

        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    return frame