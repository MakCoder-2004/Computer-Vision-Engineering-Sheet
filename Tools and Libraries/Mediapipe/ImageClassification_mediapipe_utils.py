import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

# ==========================================================
# Load Image Classifier Model
# ==========================================================
def load_image_classifier(model_path, max_results=5, score_threshold=0.0):
    """
    Load the MediaPipe Image Classifier model.
    """
    BaseOptions = mp.tasks.BaseOptions
    ImageClassifier = vision.ImageClassifier
    ImageClassifierOptions = vision.ImageClassifierOptions
    VisionRunningMode = vision.RunningMode

    options = ImageClassifierOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,  # default for video/live
        max_results=max_results,
        score_threshold=score_threshold
    )

    classifier = ImageClassifier.create_from_options(options)
    return classifier

# ==========================================================
# Classify Frame
# ==========================================================
def classify_frame(classifier, frame, timestamp_ms=None):
    """
    Perform image classification on a single frame (image or video).
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    if timestamp_ms is None:
        # IMAGE mode
        result = classifier.classify(mp_image)
    else:
        # VIDEO mode
        result = classifier.classify_for_video(mp_image, timestamp_ms)
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
# Visualize Classification Results
# ==========================================================
def visualize_classification(frame, result):
    """
    Draw classification results on the frame.
    """
    if result is None or not result.classifications:
        return frame

    overlay = frame.copy()
    y_offset = 30
    for classification in result.classifications:
        for cat in classification.categories:
            display_name = getattr(cat, "display_name", cat.category_name)
            label = f"{display_name}: {cat.score:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

            # Draw label background
            cv2.rectangle(overlay, (10, y_offset - text_height - 5),
                          (10 + text_width + 10, y_offset + 5), (0, 120, 255), -1)
            # Draw text
            cv2.putText(overlay, label, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            y_offset += text_height + 15

    # Blend overlay
    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
    return frame