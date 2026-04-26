# Computer Vision Learning Workspace

This repository organizes practical computer vision topics, notebooks, and utility code in one place.

## What This Folder Contains

- **Concepts/**
  - Introductory notebooks for core CV tasks:
    - `01.Image Classification.ipynb`
    - `02.Object Detection.ipynb`
    - `03.Image Segmentation.ipynb`
- **Tools and Libraries/**
  - Hands-on examples by framework/library:
    - `Detectron2/`
    - `EasyOCR/`
    - `Mediapipe/`
    - `OpenCv/`
    - `Yolo/`
- **Label Studio/**
  - Notes and workflow for annotation with Label Studio.
- **Models/**
  - Local model weights/checkpoints for experiments (kept out of GitHub using `.gitignore`).

## Computer Vision Topics Illustrated

This workspace demonstrates practical workflows for:

1. **Image Classification**
   - Assigning one or more labels to an image.
2. **Object Detection**
   - Locating and classifying objects using bounding boxes.
3. **Image Segmentation**
   - Pixel-level understanding (semantic/instance segmentation).
4. **Face / Pose / Hand Landmarks**
   - Keypoint extraction and geometric interpretation via Mediapipe.
5. **OCR (Optical Character Recognition)**
   - Text detection and recognition with EasyOCR.
6. **Annotation and Dataset Preparation**
   - Labeling workflows and export formats with Label Studio.

## Repository Notes

- This project is designed for learning and experimentation.
- Heavy files like model weights, generated runs, virtual environments, and IDE metadata are excluded by `.gitignore`.
- Keep notebooks and utility scripts in Git; keep binaries/artifacts local.

## Recommended GitHub Publish Flow

```bash
git init
git add .
git commit -m "Initial computer vision workspace"
# then create a GitHub repo and push
```

## License

Add your preferred license before publishing publicly (for example MIT, Apache-2.0, or CC BY-NC for educational content).
