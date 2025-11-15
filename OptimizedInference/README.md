# YOLOv8 ONNX Inference in C++

This project runs YOLOv8 inference on a video file using ONNX Runtime and OpenCV in C++.

## Dependencies

- **CMake** (>= 3.10)
- **OpenCV**
- **ONNX Runtime**

### Installation

#### OpenCV

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install libopencv-dev
```

**macOS (using Homebrew):**
```bash
brew install opencv
```

**Windows (using vcpkg):**
```bash
vcpkg install opencv4
```

#### ONNX Runtime

Download the pre-built binaries from the [ONNX Runtime releases page](https://github.com/microsoft/onnxruntime/releases). Choose the version that matches your OS and architecture.

After downloading, extract the archive and set the `ONNXRUNTIME_DIR` environment variable to the path of the extracted directory.

Example:
```bash
export ONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.13.1
```

## Build

1.  Create a build directory:
    ```bash
    mkdir build
    cd build
    ```

2.  Run CMake:
    ```bash
    cmake ..
    ```
    If CMake cannot find ONNX Runtime automatically, you can specify the path:
    ```bash
    cmake -DONNXRUNTIME_DIR=/path/to/onnxruntime ..
    ```

3.  Compile the project:
    ```bash
    make
    ```

## Run

Run the executable with the following arguments:
```bash
./yolo_inference <model_path> <video_path> <output_path>
```

-   `<model_path>`: Path to the `.onnx` model file.
-   `<video_path>`: Path to the input video file.
-   `<output_path>`: Path to save the output video with detections.

Example:
```bash
./yolo_inference ../../weights/best.onnx ../../input.mp4 ../../output.mp4
```
