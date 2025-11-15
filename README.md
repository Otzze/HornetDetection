# HornetDetection

This project contains implementations for detecting hornets in a video stream using a YOLO model.

## Python Inference

This implementation uses Python with OpenCV and Ultralytics to run YOLO inference on a video file.

### Requirements

- Python 3
- OpenCV
- Ultralytics

Install the dependencies using pip:
```bash
pip install -r requirements.txt
```

### Usage

To run the inference, execute the `YOLO_inference.py` script:
```bash
python YOLO_inference.py
```
You will need to modify the script to set the paths for the video, weights, and output.

## C++ Optimized Inference

A C++ implementation using ONNX Runtime for optimized inference is available in the `OptimizedInference` directory. This version is recommended for better performance.

For instructions on how to build and run the C++ project, please refer to the [README in the OptimizedInference directory](./OptimizedInference/README.md).
