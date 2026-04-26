import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision


# ==========================
# Load Model
# ==========================
def load_detection_model(model_path):

    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO
    )

    return vision.PoseLandmarker.create_from_options(options)


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

    if not detection_result.pose_landmarks:
        return frame

    height, width = frame.shape[:2]

    for pose_landmarks in detection_result.pose_landmarks:

        points = np.array(
            [(lm.x * width, lm.y * height) for lm in pose_landmarks],
            dtype=np.int32
        )

        for point in points:
            cv2.circle(frame, tuple(point), 4, (0, 255, 255), -1)

    return frame