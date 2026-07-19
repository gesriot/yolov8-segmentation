#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

namespace yolo_seg {

struct Yolov8Config {
    int input_width = 640;
    int input_height = 640;
    float confidence_threshold = 0.25F;
    float nms_threshold = 0.45F;
    float mask_threshold = 0.5F;
};

struct Detection {
    int class_id = -1;
    float confidence = 0.0F;
    cv::Rect box;
    cv::Mat box_mask;
};

struct StageTiming {
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    double total_ms = 0.0;
};

struct Prediction {
    std::vector<Detection> detections;
    StageTiming timing;
};

class Yolov8Segmenter {
public:
    explicit Yolov8Segmenter(const std::filesystem::path& model_path,
                             Yolov8Config config = {});

    [[nodiscard]] Prediction predict(const cv::Mat& image);

    [[nodiscard]] const Yolov8Config& config() const noexcept { return config_; }

private:
    Yolov8Config config_;
    cv::dnn::Net network_;
    std::vector<std::string> output_names_;
};

} // namespace yolo_seg
