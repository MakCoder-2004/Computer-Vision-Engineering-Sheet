import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

# ==========================================================
# Load Image Segmentation Model
# ==========================================================
def load_segmentation_model(model_path, running_mode='VIDEO', output_category_mask=True, output_confidence_masks=False):
    """
    Load the MediaPipe Image Segmenter model.
    running_mode: IMAGE, VIDEO, or LIVE_STREAM
    """
    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = vision.ImageSegmenter
    ImageSegmenterOptions = vision.ImageSegmenterOptions
    VisionRunningMode = vision.RunningMode

    mode_map = {
        'IMAGE': VisionRunningMode.IMAGE,
        'VIDEO': VisionRunningMode.VIDEO,
        'LIVE_STREAM': VisionRunningMode.LIVE_STREAM
    }

    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=mode_map.get(running_mode, VisionRunningMode.VIDEO),
        output_category_mask=output_category_mask,
        output_confidence_masks=output_confidence_masks
    )

    segmenter = ImageSegmenter.create_from_options(options)
    return segmenter

# ==========================================================
# Perform segmentation on a frame
# ==========================================================
def segment_frame(segmenter, frame, timestamp_ms=None):
    """
    Perform segmentation on a single frame.
    If timestamp_ms is None, it's IMAGE mode; otherwise VIDEO mode.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    if timestamp_ms is None:
        # IMAGE mode
        result = segmenter.segment(mp_image)
    else:
        # VIDEO mode
        result = segmenter.segment_for_video(mp_image, timestamp_ms)
    return result

# ==========================================================
# Apply mask to frame (e.g., blur background)
# ==========================================================
def apply_mask(frame, mask, blur_background=True, background_color=(0, 0, 0)):
    """
    Apply segmentation mask to an image.
    mask: uint8 array same size as frame or float confidence mask.
    If blur_background=True, background is blurred; else solid color.
    """
    if len(mask.shape) == 3:
        # If output is multiple confidence masks, take max
        mask = np.argmax(mask, axis=-1).astype(np.uint8)

    # Normalize mask to 0-1
    mask_norm = mask.astype(np.float32)
    mask_norm = cv2.normalize(mask_norm, None, 0, 1.0, cv2.NORM_MINMAX)
    mask_3ch = cv2.merge([mask_norm]*3)

    if blur_background:
        blurred = cv2.GaussianBlur(frame, (21, 21), 0)
        output = frame * mask_3ch + blurred * (1 - mask_3ch)
    else:
        bg = np.full(frame.shape, background_color, dtype=np.uint8)
        output = frame * mask_3ch + bg * (1 - mask_3ch)

    return output.astype(np.uint8)

# ==========================================================
# Display segmentation overlay
# ==========================================================
def visualize_segmentation(frame, mask, alpha=0.6, color=(0, 200, 255)):
    """
    Overlay segmentation mask on the original frame for visualization.
    """
    if len(mask.shape) == 3:
        mask = np.argmax(mask, axis=-1).astype(np.uint8)

    overlay = frame.copy()
    color_mask = np.zeros_like(frame)
    color_mask[:] = color
    overlay = cv2.addWeighted(overlay, 1.0, color_mask, alpha, 0, mask.astype(np.uint8))
    return overlay