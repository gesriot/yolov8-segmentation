#include "yolo_seg/yolov8_segmentation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

#include <opencv2/imgproc.hpp>

namespace yolo_seg {
namespace {

using Clock = std::chrono::steady_clock;

struct LetterboxResult {
    cv::Mat image;
    double scale = 1.0;
    int left = 0;
    int top = 0;
    int resized_width = 0;
    int resized_height = 0;
};

double elapsed_ms(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

LetterboxResult letterbox(const cv::Mat& source, int target_width, int target_height)
{
    LetterboxResult result;
    result.scale = std::min(
        static_cast<double>(target_width) / source.cols,
        static_cast<double>(target_height) / source.rows);
    result.resized_width = static_cast<int>(std::round(source.cols * result.scale));
    result.resized_height = static_cast<int>(std::round(source.rows * result.scale));

    cv::Mat resized;
    if (source.cols != result.resized_width || source.rows != result.resized_height) {
        cv::resize(source, resized, cv::Size(result.resized_width, result.resized_height));
    }
    else {
        resized = source;
    }

    const double half_width = (target_width - result.resized_width) / 2.0;
    const double half_height = (target_height - result.resized_height) / 2.0;
    result.left = static_cast<int>(std::round(half_width - 0.1));
    result.top = static_cast<int>(std::round(half_height - 0.1));
    const int right = static_cast<int>(std::round(half_width + 0.1));
    const int bottom = static_cast<int>(std::round(half_height + 0.1));

    cv::copyMakeBorder(resized, result.image, result.top, bottom, result.left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return result;
}

// YOLOv8 boxes are [cx, cy, w, h] in letterbox space.
cv::Rect map_box_to_source(const float* row, const LetterboxResult& transform,
                           const cv::Size& source_size)
{
    const double x1 = (row[0] - row[2] / 2.0 - transform.left) / transform.scale;
    const double y1 = (row[1] - row[3] / 2.0 - transform.top) / transform.scale;
    const double x2 = (row[0] + row[2] / 2.0 - transform.left) / transform.scale;
    const double y2 = (row[1] + row[3] / 2.0 - transform.top) / transform.scale;

    const int left = std::clamp(static_cast<int>(std::floor(x1)), 0, source_size.width);
    const int top = std::clamp(static_cast<int>(std::floor(y1)), 0, source_size.height);
    const int right = std::clamp(static_cast<int>(std::ceil(x2)), 0, source_size.width);
    const int bottom = std::clamp(static_cast<int>(std::ceil(y2)), 0, source_size.height);
    if (right <= left || bottom <= top) {
        return {};
    }
    return {left, top, right - left, bottom - top};
}

void validate_outputs(const std::vector<cv::Mat>& outputs)
{
    if (outputs.size() != 2) {
        throw std::runtime_error("YOLOv8 requires exactly two output tensors");
    }
    if (outputs[0].dims != 3 || outputs[0].size[0] != 1) {
        throw std::runtime_error("Unexpected YOLOv8 detection tensor shape");
    }
    if (outputs[1].dims != 4 || outputs[1].size[0] != 1) {
        throw std::runtime_error("Unexpected YOLOv8 prototype tensor shape");
    }
    if (outputs[0].size[1] <= 4 + outputs[1].size[1]) {
        throw std::runtime_error("Detection rows are too short for box, scores and mask coefficients");
    }
}

cv::Mat reconstruct_box_mask(const float* coefficients, const cv::Mat& prototypes,
                             const LetterboxResult& transform, const cv::Size& source_size,
                             const cv::Rect& box, const Yolov8Config& config)
{
    const int channels = prototypes.size[1];
    const int proto_height = prototypes.size[2];
    const int proto_width = prototypes.size[3];

    cv::Mat coefficient_matrix(1, channels, CV_32F,
                               const_cast<float*>(coefficients));
    cv::Mat prototype_matrix(channels, proto_height * proto_width, CV_32F,
                             const_cast<float*>(prototypes.ptr<float>()));
    cv::Mat logits = coefficient_matrix * prototype_matrix;
    logits = logits.reshape(1, proto_height);

    cv::Mat negative_exp;
    cv::exp(-logits, negative_exp);
    cv::Mat probabilities = 1.0 / (1.0 + negative_exp);

    cv::Mat input_mask;
    cv::resize(probabilities, input_mask,
               cv::Size(config.input_width, config.input_height), 0.0, 0.0,
               cv::INTER_LINEAR);

    const cv::Rect content_rect(
        transform.left, transform.top, transform.resized_width, transform.resized_height);
    cv::Mat source_mask;
    cv::resize(input_mask(content_rect), source_mask, source_size, 0.0, 0.0,
               cv::INTER_LINEAR);

    cv::Mat box_mask = source_mask(box) > config.mask_threshold;
    return box_mask;
}

} // namespace

Yolov8Segmenter::Yolov8Segmenter(const std::filesystem::path& model_path,
                                 Yolov8Config config)
    : config_(config),
      network_(cv::dnn::readNet(model_path.string()))
{
    if (config_.input_width <= 0 || config_.input_height <= 0) {
        throw std::invalid_argument("YOLOv8 input dimensions must be positive");
    }
    network_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    network_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    output_names_ = network_.getUnconnectedOutLayersNames();
    if (output_names_.size() != 2) {
        throw std::runtime_error("YOLOv8 ONNX model must expose two outputs");
    }
}

Prediction Yolov8Segmenter::predict(const cv::Mat& image)
{
    if (image.empty() || image.type() != CV_8UC3) {
        throw std::invalid_argument("YOLOv8 input must be a non-empty CV_8UC3 image");
    }

    Prediction prediction;
    const auto total_start = Clock::now();

    const auto preprocess_start = Clock::now();
    const LetterboxResult transformed =
        letterbox(image, config_.input_width, config_.input_height);
    cv::Mat blob = cv::dnn::blobFromImage(
        transformed.image, 1.0 / 255.0,
        cv::Size(config_.input_width, config_.input_height), {}, true, false);
    const auto preprocess_end = Clock::now();

    const auto inference_start = preprocess_end;
    network_.setInput(blob);
    std::vector<cv::Mat> outputs;
    network_.forward(outputs, output_names_);
    const auto inference_end = Clock::now();

    const auto postprocess_start = inference_end;
    validate_outputs(outputs);
    const cv::Mat& detection_tensor = outputs[0];
    const cv::Mat& prototypes = outputs[1];
    const int row_width = detection_tensor.size[1];
    const int class_count = row_width - 4 - prototypes.size[1];

    // [1, 4+classes+coeffs, anchors] -> [anchors, 4+classes+coeffs] with
    // contiguous rows.
    const cv::Mat rows = cv::Mat(row_width, detection_tensor.size[2], CV_32F,
                                 const_cast<float*>(detection_tensor.ptr<float>()))
                             .t();

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<int> row_indices;
    for (int row_index = 0; row_index < rows.rows; ++row_index) {
        const float* row = rows.ptr<float>(row_index);
        const cv::Mat scores(1, class_count, CV_32F, const_cast<float*>(row + 4));
        cv::Point class_point;
        double class_score = 0.0;
        cv::minMaxLoc(scores, nullptr, &class_score, nullptr, &class_point);
        if (!std::isfinite(class_score) || class_score < config_.confidence_threshold) {
            continue;
        }

        const cv::Rect box = map_box_to_source(row, transformed, image.size());
        if (box.empty()) {
            continue;
        }
        class_ids.push_back(class_point.x);
        confidences.push_back(static_cast<float>(class_score));
        boxes.push_back(box);
        row_indices.push_back(row_index);
    }

    std::vector<int> kept;
    cv::dnn::NMSBoxes(boxes, confidences, config_.confidence_threshold,
                      config_.nms_threshold, kept);

    prediction.detections.reserve(kept.size());
    for (const int index : kept) {
        Detection detection;
        detection.class_id = class_ids[index];
        detection.confidence = confidences[index];
        detection.box = boxes[index];
        detection.box_mask = reconstruct_box_mask(
            rows.ptr<float>(row_indices[index]) + 4 + class_count, prototypes,
            transformed, image.size(), detection.box, config_);
        prediction.detections.push_back(std::move(detection));
    }
    const auto postprocess_end = Clock::now();

    prediction.timing.preprocess_ms = elapsed_ms(preprocess_start, preprocess_end);
    prediction.timing.inference_ms = elapsed_ms(inference_start, inference_end);
    prediction.timing.postprocess_ms = elapsed_ms(postprocess_start, postprocess_end);
    prediction.timing.total_ms = elapsed_ms(total_start, postprocess_end);
    return prediction;
}

} // namespace yolo_seg
