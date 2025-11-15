cmake_minimum_required(VERSION 3.10)
project(YoloInference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV and set OpenCV_DIR.")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

# Find ONNX Runtime
# You might need to set ONNXRUNTIME_DIR to the path of your ONNX Runtime installation
# e.g., cmake -DONNXRUNTIME_DIR=/path/to/onnxruntime ..

if(NOT onnxruntime_FOUND)
    message(WARNING "onnxruntime package not found. Trying to find it manually.")
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
              HINTS ENV ONNXRUNTIME_DIR ${ONNXRUNTIME_DIR}
              PATH_SUFFIXES include)
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
                 HINTS ENV ONNXRUNTIME_DIR ${ONNXRUNTIME_DIR}
                 PATH_SUFFIXES lib lib64)

    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
        set(ONNXRUNTIME_FOUND TRUE)
        include_directories(${ONNXRUNTIME_INCLUDE_DIR})
        message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIBRARY}")
    else()
        message(FATAL_ERROR "ONNX Runtime not found. Please install it and set the ONNXRUNTIME_DIR environment variable.")
    endif()
else()
    message(STATUS "Found ONNX Runtime package")
    include_directories(${onnxruntime_INCLUDE_DIRS})
    set(ONNXRUNTIME_LIBRARY onnxruntime::onnxruntime)
endif()


add_executable(yolo_inference main.cpp)

target_link_libraries(yolo_inference ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARY})

