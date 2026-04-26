# Label Studio

## Overview
Label Studio is an open-source data labeling tool that helps you annotate various types of data including images, audio, text, time series, and multi-domain data. It provides a flexible interface for creating high-quality training data for machine learning models.

## Getting Started

### Access Label Studio
Once Label Studio is running, you can access it through your web browser at:

**[http://localhost:8080/](http://localhost:8080/)**

### Starting Label Studio
To start Label Studio, run the following command in your terminal:
```bash
label-studio start
```

By default, it will run on port 8080. If you need to use a different port, use:
```bash
label-studio start --port <port_number>
```

### Initial Setup
1. Open [http://localhost:8080/](http://localhost:8080/) in your web browser
2. Create an account (stored locally)
3. Create a new project
4. Configure your labeling interface
5. Import your data
6. Start annotating!

## Annotation Types and Use Cases

Different annotation types are used for different computer vision tasks. Here's a comprehensive overview:

| Annotation Type | Task                  | Example Models     |
| --------------- | --------------------- | ------------------ |
| Image Label     | Classification        | ResNet, VGG        |
| Bounding Box    | Object Detection      | YOLO, Faster R-CNN |
| Polygon         | Instance Segmentation | Mask R-CNN         |
| Pixel-Level     | Semantic Segmentation | U-Net, DeepLab     |
| Keypoints       | Pose Estimation       | OpenPose           |
| 3D Bounding Box | 3D Detection          | PointNet           |
| Binary Mask     | Background Removal    | U-Net              |
| Optical Flow    | Motion Estimation     | RAFT               |

## Export Formats

Label Studio supports multiple export formats to work seamlessly with various machine learning frameworks and models:

| Format     | File Type  | Mainly Used By         |
| ---------- | ---------- | ---------------------- |
| Pascal VOC | XML        | Faster R-CNN, SSD      |
| YOLO       | TXT        | YOLOv3–v9              |
| COCO       | JSON       | Mask R-CNN, Detectron2 |
| TFRecord   | TFRecord   | TensorFlow OD API      |
| PNG Mask   | Image file | U-Net, DeepLab         |
| CSV        | CSV        | Custom pipelines       |

## Key Features

- **Multi-format Support**: Import images, videos, audio, text, HTML, and time-series data
- **Flexible Configuration**: Customize the labeling interface using XML-like tags
- **Team Collaboration**: Multiple users can work on the same project
- **ML Integration**: Connect with machine learning models for pre-annotation and active learning
- **Quality Control**: Review and validate annotations
- **Export Options**: Export in various formats compatible with popular ML frameworks

## Common Workflows

### Object Detection (YOLO)
1. Create a project with bounding box annotation
2. Import your images
3. Draw bounding boxes around objects
4. Export in YOLO format (.txt files)

### Image Segmentation
1. Create a project with polygon or brush annotation
2. Import your images
3. Draw polygons or use brush to mark regions
4. Export in COCO or PNG Mask format

### Image Classification
1. Create a project with image classification
2. Import your images
3. Assign labels to each image
4. Export in CSV or JSON format

## Tips

- Use keyboard shortcuts to speed up annotation
- Set up quality control with multiple annotators
- Use pre-annotations from existing models to speed up the process
- Regularly export your work to avoid data loss
- Use filters and views to organize large datasets

## Resources

- [Official Documentation](https://labelstud.io/guide/)
- [GitHub Repository](https://github.com/heartexlabs/label-studio)
- [Community Forum](https://github.com/heartexlabs/label-studio/discussions)

## Troubleshooting

### Port Already in Use
If port 8080 is already in use, specify a different port:
```bash
label-studio start --port 8090
```

### Can't Access the Interface
- Ensure Label Studio is running (check terminal output)
- Try accessing `http://127.0.0.1:8080/` instead
- Check firewall settings
- Clear browser cache and cookies

### Data Not Saving
- Check disk space
- Ensure you have write permissions in the Label Studio directory
- Check the terminal for error messages

---

**Note**: Label Studio stores data locally by default. Make sure to export and backup your annotations regularly.

