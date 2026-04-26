import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe_model_maker import gesture_recognizer

# ==========================================================
# Load Gesture Recognition Model
# ==========================================================
def load_gesture_model(model_path, num_hands=2):
    """
    Load a pre-trained MediaPipe Gesture Recognizer model.
    """
    GestureRecognizerOptions = vision.GestureRecognizerOptions
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = vision.RunningMode
    GestureRecognizer = vision.GestureRecognizer

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )

    recognizer = GestureRecognizer.create_from_options(options)
    return recognizer

# ==========================================================
# Detect Gestures in Frame
# ==========================================================
def detect_gesture(recognizer, frame, timestamp_ms):
    """
    Perform gesture recognition on a single video frame.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = recognizer.recognize_for_video(mp_image, timestamp_ms)
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
# Visualize Gesture Recognition Results
# ==========================================================
def visualize_gesture(frame, result):
    """
    Draw hand landmarks, bounding boxes, and gesture labels on the frame.
    """
    if result is None or not result.hand_landmarks:
        return frame

    height, width = frame.shape[:2]

    for idx, hand_landmarks in enumerate(result.hand_landmarks):
        # Convert normalized landmarks to pixel coordinates
        landmarks = np.array(
            [(lm.x * width, lm.y * height) for lm in hand_landmarks],
            dtype=np.int32
        )

        # Draw skeleton
        connections = [
            (0,1),(1,2),(2,3),(3,4),      # Thumb
            (0,5),(5,6),(6,7),(7,8),      # Index
            (0,9),(9,10),(10,11),(11,12), # Middle
            (0,13),(13,14),(14,15),(15,16), # Ring
            (0,17),(17,18),(18,19),(19,20)  # Pinky
        ]
        for start_idx, end_idx in connections:
            cv2.line(frame, tuple(landmarks[start_idx]), tuple(landmarks[end_idx]), (0,255,0), 2)

        for point in landmarks:
            cv2.circle(frame, tuple(point), 4, (0,0,255), -1)

        # Bounding box
        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)

        # Handedness label
        handedness = result.handedness[idx][0].category_name
        score = result.handedness[idx][0].score

        # Gesture label
        gesture_name = result.gestures[idx][0].category_name
        gesture_score = result.gestures[idx][0].score

        label = f"{handedness} {score:.2f} | {gesture_name} {gesture_score:.2f}"

        # Draw rectangle
        overlay = frame.copy()
        draw_rounded_rectangle(overlay, (x_min, y_min), (x_max, y_max), (255, 150, 0), 3, 15)
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

        # Draw label background
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x_min, y_min - 30), (x_min + text_width + 10, y_min), (255,150,0), -1)
        cv2.putText(frame, label, (x_min + 5, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    return frame

# ==========================================================
# Custom Training Section (MediaPipe Model Maker)
# ==========================================================
def train_custom_gesture_model(dataset_path, export_dir="exported_model",
                               learning_rate=0.001, batch_size=2, epochs=10,
                               dropout_rate=0.05, layer_widths=[]):
    """
    Train a custom gesture recognizer using MediaPipe Model Maker.
    Dataset folder format: dataset_path/<label_name>/<img_name>.*

    Returns the trained model object.
    """
    # Load dataset
    data = gesture_recognizer.Dataset.from_folder(
        dirname=dataset_path,
        hparams=gesture_recognizer.HandDataPreprocessingParams()
    )
    train_data, rest_data = data.split(0.8)
    validation_data, test_data = rest_data.split(0.5)

    # Hyperparameters
    hparams = gesture_recognizer.HParams(
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        export_dir=export_dir
    )

    # Model options
    model_options = gesture_recognizer.ModelOptions(
        dropout_rate=dropout_rate,
        layer_widths=layer_widths
    )

    # Create & train model
    options = gesture_recognizer.GestureRecognizerOptions(
        model_options=model_options,
        hparams=hparams
    )
    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options
    )

    # Evaluate
    loss, accuracy = model.evaluate(test_data)
    print(f"Test loss: {loss}, Test accuracy: {accuracy}")

    # Export TFLite model for on-device use
    model.export_model()
    return model