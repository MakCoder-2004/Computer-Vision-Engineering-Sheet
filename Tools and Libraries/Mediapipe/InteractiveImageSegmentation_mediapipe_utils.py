import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

# ==========================================================
# Load Interactive Image Segmentation Model
# ==========================================================
def load_interactive_segmenter(model_path, running_mode='IMAGE', output_category_mask=True, output_confidence_masks=False):
    """
    Load the MediaPipe Interactive Image Segmenter model.
    running_mode: IMAGE, VIDEO, or LIVE_STREAM
    """
    BaseOptions = mp.tasks.BaseOptions
    InteractiveSegmenter = vision.InteractiveSegmenter
    InteractiveSegmenterOptions = vision.InteractiveSegmenterOptions
    VisionRunningMode = vision.RunningMode

    mode_map = {
        'IMAGE': VisionRunningMode.IMAGE,
        'VIDEO': VisionRunningMode.VIDEO,
        'LIVE_STREAM': VisionRunningMode.LIVE_STREAM
    }

    options = InteractiveSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=mode_map.get(running_mode, VisionRunningMode.IMAGE),
        output_category_mask=output_category_mask,
        output_confidence_masks=output_confidence_masks
    )

    segmenter = InteractiveSegmenter.create_from_options(options)
    return segmenter

# ==========================================================
# Create Region of Interest
# ==========================================================
def create_roi(x, y):
    """
    Create a RegionOfInterest for the interactive segmenter.
    x, y: normalized coordinates [0,1] relative to image width/height
    """
    RegionOfInterest = vision.InteractiveSegmenterRegionOfInterest
    NormalizedKeypoint = vision.NormalizedKeypoint

    roi = RegionOfInterest(
        format=RegionOfInterest.Format.KEYPOINT,
        keypoint=NormalizedKeypoint(x=x, y=y)
    )
    return roi

# ==========================================================
# Segment Image at ROI
# ==========================================================
def segment_at_point(segmenter, frame, x, y):
    """
    Perform interactive segmentation at a specific normalized point (x, y).
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    roi = create_roi(x, y)
    result = segmenter.segment(mp_image, roi)
    return result

# ==========================================================
# Apply mask to frame (e.g., blur background)
# ==========================================================
def apply_mask(frame, mask, blur_background=True, background_color=(0, 0, 0)):
    """
    Apply interactive segmentation mask to an image.
    mask: uint8 array or float confidence mask
    """
    if len(mask.shape) == 3:
        # If output is multiple confidence masks, take max
        mask = np.argmax(mask, axis=-1).astype(np.uint8)

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
# Visualize interactive segmentation
# ==========================================================
def visualize_segmentation(frame, mask, alpha=0.6, color=(0, 200, 255)):
    """
    Overlay interactive segmentation mask on the original frame.
    """
    if len(mask.shape) == 3:
        mask = np.argmax(mask, axis=-1).astype(np.uint8)

    overlay = frame.copy()
    color_mask = np.zeros_like(frame)
    color_mask[:] = color
    overlay = cv2.addWeighted(overlay, 1.0, color_mask, alpha, 0, mask.astype(np.uint8))
    return overlay