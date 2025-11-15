// main.cpp
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cstring>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <iomanip>

// Struct to hold detection results
struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

void print_progress(size_t current, size_t total) {
    const int bar_width = 40;
    float progress = (float)current / total;

    std::cout << "\r[";
    int pos = bar_width * progress;
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "#";
        else std::cout << "-";
    }

    std::cout << "] " 
              << std::fixed << std::setprecision(1) 
              << (progress * 100.0) << "%  (" 
              << current << " / " << total << " frames)" << std::flush;
}

// Preprocess: resize, BGR->RGB, normalize, produce CHW float vector
// input_shape is expected to be something like {1, 3, H, W} but may contain -1 (dynamic dims)
std::vector<float> preprocess(const cv::Mat& frame, const std::vector<int64_t>& input_shape, int& out_h, int& out_w) {
    // Determine target H/W from input_shape, fallback to 640 if dynamic
    int target_h = 640;
    int target_w = 640;
    if (input_shape.size() >= 4) {
        if (input_shape[2] > 0) target_h = static_cast<int>(input_shape[2]);
        if (input_shape[3] > 0) target_w = static_cast<int>(input_shape[3]);
    }

    out_h = target_h;
    out_w = target_w;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(target_w, target_h));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC3, 1.0f / 255.0f);

    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels); // R, G, B

    size_t plane_size = static_cast<size_t>(target_h) * static_cast<size_t>(target_w);
    std::vector<float> chw;
    chw.resize(3 * plane_size);

    // Copy channel data into chw vector in R,G,B channel order -> CHW
    // OpenCV stores channel Mat as continuous floats
    std::memcpy(chw.data() + 0 * plane_size, channels[0].data, plane_size * sizeof(float));
    std::memcpy(chw.data() + 1 * plane_size, channels[1].data, plane_size * sizeof(float));
    std::memcpy(chw.data() + 2 * plane_size, channels[2].data, plane_size * sizeof(float));

    return chw;
}

// Postprocessing: supports outputs laid out as [1, num_preds, attrs] or [1, attrs, num_preds].
// attrs = 4 + num_classes (x,y,w,h + class probs)
std::vector<Detection> postprocess(const Ort::Value& output_tensor,
                                   const cv::Size& original_frame_size,
                                   const cv::Size& resized_frame_size,
                                   float conf_threshold = 0.5f) {
    const float* raw_output = output_tensor.GetTensorData<float>();
    std::vector<int64_t> output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();

    std::vector<Detection> detections;

    if (output_shape.size() != 3) {
        std::cerr << "Unexpected output tensor shape (expected 3 dims), got: ";
        for (auto d : output_shape) std::cerr << d << " ";
        std::cerr << std::endl;
        return detections;
    }

    int64_t dim1 = output_shape[1];
    int64_t dim2 = output_shape[2];
    int64_t num_preds = 0;
    int64_t attrs = 0;
    bool layout_is_NxA = false; // true if shape is [1, num_preds, attrs]

    if (dim2 > dim1) {
        // Most common: [1, num_preds, attrs]
        num_preds = dim1;
        attrs = dim2;
        layout_is_NxA = true;
    } else {
        // Other common export: [1, attrs, num_preds]
        attrs = dim1;
        num_preds = dim2;
        layout_is_NxA = false;
    }

    if (attrs < 5) {
        std::cerr << "Unexpected attrs < 5, can't parse (attrs=" << attrs << ")" << std::endl;
        return detections;
    }

    int num_classes = static_cast<int>(attrs - 4);

    // scaling from resized->original
    float scale_x = static_cast<float>(original_frame_size.width) / static_cast<float>(resized_frame_size.width);
    float scale_y = static_cast<float>(original_frame_size.height) / static_cast<float>(resized_frame_size.height);

    // iterate predictions
    for (int64_t i = 0; i < num_preds; ++i) {
        // get pointer to first attribute for detection i depending on layout
        // index calculation:
        // if layout_is_NxA: index = i * attrs + j
        // else: index = j * num_preds + i
        auto get_val = [&](int64_t j)->float {
            if (layout_is_NxA) {
                return raw_output[i * attrs + j];
            } else {
                return raw_output[j * num_preds + i];
            }
        };

        // first 4 are box: cx, cy, w, h
        float cx = get_val(0);
        float cy = get_val(1);
        float w = get_val(2);
        float h = get_val(3);

        // find class with max probability
        float max_conf = 0.0f;
        int class_id = -1;
        for (int c = 0; c < num_classes; ++c) {
            float conf = get_val(4 + c);
            if (conf > max_conf) {
                max_conf = conf;
                class_id = c;
            }
        }

        if (max_conf > conf_threshold) {
            // Convert center format to box on resized image coordinates
            // Assumes cx,cy,w,h are in pixel coordinates relative to resized image,
            // or normalized depending on your export. This code assumes they are relative to resized image.
            int left = static_cast<int>((cx - 0.5f * w) * scale_x);
            int top = static_cast<int>((cy - 0.5f * h) * scale_y);
            int width = static_cast<int>(w * scale_x);
            int height = static_cast<int>(h * scale_y);

            // Clamp
            left = std::max(0, std::min(left, original_frame_size.width - 1));
            top = std::max(0, std::min(top, original_frame_size.height - 1));
            if (left + width > original_frame_size.width) width = original_frame_size.width - left;
            if (top + height > original_frame_size.height) height = original_frame_size.height - top;

            if (width > 0 && height > 0) {
                detections.push_back({cv::Rect(left, top, width, height), max_conf, class_id});
            }
        }
    }

    return detections;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model_path.onnx> <video_path> <output_path>" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string video_path = argv[2];
    std::string output_path = argv[3];

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YoloInference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session session(env, model_path.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // Get input and output counts
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();

    if (num_input_nodes == 0) {
        std::cerr << "Model has no inputs." << std::endl;
        return -1;
    }

    // Get input names and shapes (use Allocated name to keep memory safe)
    std::vector<const char*> input_node_names;
    std::vector<std::string> input_name_storage;
    std::vector<std::vector<int64_t>> input_node_dims;

    for (size_t i = 0; i < num_input_nodes; ++i) {
        auto name_ptr = session.GetInputNameAllocated(i, allocator);
        input_name_storage.emplace_back(name_ptr.get());
        input_node_names.push_back(input_name_storage.back().c_str());

        auto type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_node_dims.push_back(tensor_info.GetShape());
    }

    // Get output names (kept alive in storage)
    std::vector<const char*> output_node_names;
    std::vector<std::string> output_name_storage;
    for (size_t i = 0; i < num_output_nodes; ++i) {
        auto name_ptr = session.GetOutputNameAllocated(i, allocator);
        output_name_storage.emplace_back(name_ptr.get());
        output_node_names.push_back(output_name_storage.back().c_str());
    }

    // Open video
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file: " << video_path << std::endl;
        return -1;
    }

    // Video writer
    int codec = cv::VideoWriter::fourcc('m','p','4','v');
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 25.0;
    cv::Size frame_size(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    cv::VideoWriter writer;
    writer.open(output_path, codec, fps, frame_size, true);
    if (!writer.isOpened()) {
        std::cerr << "Failed to open writer for: " << output_path << std::endl;
        return -1;
    }

    // Pre-alloc memory info for inputs
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    cv::Mat frame;

    size_t total_frames = static_cast<size_t>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    size_t current_frame = 0;

    while (cap.read(frame)) {
        current_frame++;

        print_progress(current_frame, total_frames);

        // Get the model input shape template
        std::vector<int64_t> in_shape = input_node_dims[0]; // e.g. {1,3,H,W} or {1,3,-1,-1}
        // Make a copy and replace dynamic dims with concrete sizes after preprocess
        int target_h = 0, target_w = 0;
        std::vector<float> input_tensor_values = preprocess(frame, in_shape, target_h, target_w);

        // ensure shape is 4D: batch, channel, H, W
        std::vector<int64_t> tensor_shape = in_shape;
        if (tensor_shape.size() < 4) {
            // fallback to {1,3,H,W}
            tensor_shape = {1, 3, target_h, target_w};
        } else {
            // replace -1 or <=0 dims
            if (tensor_shape[0] <= 0) tensor_shape[0] = 1;
            if (tensor_shape[1] <= 0) tensor_shape[1] = 3;
            if (tensor_shape[2] <= 0) tensor_shape[2] = target_h;
            if (tensor_shape[3] <= 0) tensor_shape[3] = target_w;
        }

        // create input tensor
        size_t input_tensor_size = 1;
        for (auto d : tensor_shape) input_tensor_size *= static_cast<size_t>(d);

        if (input_tensor_values.size() != input_tensor_size) {
            // If mismatch, attempt to adapt: some models expect NHWC or different layouts.
            std::cerr << "Input tensor size mismatch: expected " << input_tensor_size
                      << " floats, got " << input_tensor_values.size() << ". Aborting frame." << std::endl;
            continue;
        }

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_size,
            tensor_shape.data(),
            static_cast<int>(tensor_shape.size())
        );

        // Run inference
        std::array<Ort::Value, 1> input_tensors_arr = { std::move(input_tensor) };
        auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                          input_node_names.data(), // const char* const*
                                          input_tensors_arr.data(), // Ort::Value*
                                          1, // input count
                                          output_node_names.data(),
                                          static_cast<int>(output_node_names.size()));

        if (output_tensors.empty()) {
            std::cerr << "Model returned no outputs." << std::endl;
            continue;
        }

        // Postprocess - we use resized_frame_size same as target_h/target_w
        cv::Size resized_size(target_w, target_h);
        auto detections = postprocess(output_tensors[0], frame_size, resized_size, 0.5f);

        // Draw boxes
        for (const auto& d : detections) {
            cv::rectangle(frame, d.box, cv::Scalar(0, 255, 0), 2);
            std::string label = std::to_string(d.class_id) + ": " + std::to_string(d.confidence);
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::putText(frame, label, cv::Point(std::max(0, d.box.x), std::max(0, d.box.y - 5)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
        }

        writer.write(frame);
    }

    std::cout << std::endl;
    cap.release();
    writer.release();

    std::cout << "Inference complete. Output saved to " << output_path << std::endl;
    return 0;
}

