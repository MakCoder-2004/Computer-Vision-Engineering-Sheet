import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks.python import vision

# ==========================================================
# Load Detection Model
# ==========================================================
def load_detection_model(model_path):

    HandLandmarkerOptions = vision.HandLandmarkerOptions
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = vision.RunningMode
    HandLandmarker = vision.HandLandmarker

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )

    landmarker = HandLandmarker.create_from_options(options)
    return landmarker

# ==========================================================
# Detect Frame
# ==========================================================
def detect_frame(landmarker, frame, timestamp_ms):

    # Convert BGR (OpenCV) → RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    return result

# ==========================================================
# Visualizing Frame
# ==========================================================
def draw_rounded_rectangle(img, top_left, bottom_right, color, thickness=2, radius=15):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def visualize_detection(frame, detection_result):

    if detection_result is None or not detection_result.hand_landmarks:
        return frame

    height, width = frame.shape[:2]

    FINGER_CONNECTIONS = {
        "Thumb": {"connections": [(0,1),(1,2),(2,3),(3,4)], "color": (0,0,255)},
        "Index": {"connections": [(0,5),(5,6),(6,7),(7,8)], "color": (255,0,0)},
        "Middle": {"connections": [(0,9),(9,10),(10,11),(11,12)], "color": (0,255,0)},
        "Ring": {"connections": [(0,13),(13,14),(14,15),(15,16)], "color": (0,255,255)},
        "Pinky": {"connections": [(0,17),(17,18),(18,19),(19,20)], "color": (255,0,255)},
        "Palm": {"connections": [(5,9),(9,13),(13,17)], "color": (200,200,200)}
    }

    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):

        landmarks = np.array(
            [(lm.x * width, lm.y * height) for lm in hand_landmarks],
            dtype=np.int32
        )

        # =========================
        # Draw Colored Fingers
        # =========================
        for finger_data in FINGER_CONNECTIONS.values():
            color = finger_data["color"]
            for start_idx, end_idx in finger_data["connections"]:
                cv2.line(
                    frame,
                    tuple(landmarks[start_idx]),
                    tuple(landmarks[end_idx]),
                    color,
                    3
                )

        for point in landmarks:
            cv2.circle(frame, tuple(point), 4, (255,255,255), -1)

        # =========================
        # Draw BBox + Label
        # =========================
        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)

        padding = 25
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)

        handedness = detection_result.handedness[idx][0].category_name
        score = detection_result.handedness[idx][0].score

        label = f"{handedness}  {score:.2f}"

        # Color theme per hand
        if handedness == "Left":
            box_color = (255, 120, 0)
        else:
            box_color = (255, 180, 0)

        # Transparent overlay
        overlay = frame.copy()
        draw_rounded_rectangle(overlay,
                               (x_min, y_min),
                               (x_max, y_max),
                               box_color,
                               thickness=3,
                               radius=20)

        alpha = 0.85
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Label background
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )

        label_x1 = x_min
        label_y1 = y_min - 40
        label_x2 = x_min + text_width + 20
        label_y2 = y_min

        cv2.rectangle(frame,
                      (label_x1, label_y1),
                      (label_x2, label_y2),
                      box_color,
                      -1)

        cv2.putText(frame,
                    label,
                    (label_x1 + 10, label_y2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,255),
                    2)

    return frame