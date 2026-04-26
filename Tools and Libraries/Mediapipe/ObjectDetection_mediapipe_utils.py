import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

# ==========================================================
# Load Object Detector Model
# ==========================================================
def load_object_detector(model_path, max_results=5, score_threshold=0.0):
    """
    Load the MediaPipe Object Detector model.
    """
    BaseOptions = mp.tasks.BaseOptions
    ObjectDetector = vision.ObjectDetector
    ObjectDetectorOptions = vision.ObjectDetectorOptions
    VisionRunningMode = vision.RunningMode

    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,  # default for video/live
        max_results=max_results,
        score_threshold=score_threshold
    )

    detector = ObjectDetector.create_from_options(options)
    return detector

# ==========================================================
# Detect Objects in Frame
# ==========================================================
def detect_objects(detector, frame, timestamp_ms=None):
    """
    Perform object detection on a single frame (image or video).
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    if timestamp_ms is None:
        # IMAGE mode
        result = detector.detect(mp_image)
    else:
        # VIDEO mode
        result = detector.detect_for_video(mp_image, timestamp_ms)
    return result

# ==========================================================
# Draw Rounded Rectangle Helper
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

# ==========================================================
# Visualize Detection Results
# ==========================================================
def visualize_detections(frame, result):
    """
    Draw bounding boxes and labels for each detected object.
    """
    if result is None or not result.detections:
        return frame

    overlay = frame.copy()

    height, width = frame.shape[:2]

    for detection in result.detections:
        # Get bounding box in pixel coordinates
        bbox = detection.bounding_box
        x_min = int(bbox.origin_x)
        y_min = int(bbox.origin_y)
        x_max = int(bbox.origin_x + bbox.width)
        y_max = int(bbox.origin_y + bbox.height)

        # Draw rounded rectangle
        draw_rounded_rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 200, 255), 3, 15)

        # Get label (category with highest score)
        if detection.categories:
            cat = detection.categories[0]
            label_name = getattr(cat, "display_name", cat.category_name)
            score = cat.score
            label = f"{label_name}: {score:.2f}"

            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(overlay, (x_min, y_min - text_height - 5),
                          (x_min + text_width + 10, y_min), (0, 200, 255), -1)
            # Draw text
            cv2.putText(overlay, label, (x_min + 5, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Blend overlay with frame
    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
    return frame