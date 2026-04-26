import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision


def load_detection_model(model_path):

    options = vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO
    )

    return vision.FaceDetector.create_from_options(options)


def detect_frame(detector, frame, timestamp_ms):

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    return detector.detect_for_video(mp_image, timestamp_ms)


def visualize_detection(frame, detection_result):

    if not detection_result.detections:
        return frame

    for detection in detection_result.detections:

        bbox = detection.bounding_box
        x, y = bbox.origin_x, bbox.origin_y
        w, h = bbox.width, bbox.height

        cv2.rectangle(frame,
                      (x, y),
                      (x + w, y + h),
                      (0, 255, 0),
                      2)

    return frame