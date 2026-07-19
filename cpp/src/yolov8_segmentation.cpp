#include "yolo_seg/yolov8_segmentation.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace yolo_seg {
namespace {

using Clock = std::chrono::steady_clock;

struct LetterboxResult {
    cv::Mat image;
    double scale = 1.0;
    int left = 0;
    int top = 0;
};

[[nodiscard]] double elapsed_ms(Clock::time_point start, Clock::time_point end) noexcept
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

[[nodiscard]] LetterboxResult letterbox(const cv::Mat& source, int target_width,
                                        int target_height)
{
    LetterboxResult result;
    result.scale = std::min(
        static_cast<double>(target_width) / source.cols,
        static_cast<double>(target_height) / source.rows);
    const int resized_width = static_cast<int>(std::round(source.cols * result.scale));
    const int resized_height = static_cast<int>(std::round(source.rows * result.scale));

    cv::Mat resized;
    if (source.cols != resized_width || source.rows != resized_height) {
        cv::resize(source, resized, cv::Size(resized_width, resized_height));
    }
    else {
        resized = source;
    }

    const double half_width = (target_width - resized_width) / 2.0;
    const double half_height = (target_height - resized_height) / 2.0;
    result.left = static_cast<int>(std::round(half_width - 0.1));
    result.top = static_cast<int>(std::round(half_height - 0.1));
    const int right = static_cast<int>(std::round(half_width + 0.1));
    const int bottom = static_cast<int>(std::round(half_height + 0.1));

    cv::copyMakeBorder(resized, result.image, result.top, bottom, result.left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return result;
}

// YOLOv8 boxes are [cx, cy, w, h] in letterbox space.
[[nodiscard]] cv::Rect map_box_to_source(const std::array<float, 4>& box,
                                         const LetterboxResult& transform,
                                         const cv::Size& source_size) noexcept
{
    const double x1 = (box[0] - box[2] / 2.0 - transform.left) / transform.scale;
    const double y1 = (box[1] - box[3] / 2.0 - transform.top) / transform.scale;
    const double x2 = (box[0] + box[2] / 2.0 - transform.left) / transform.scale;
    const double y2 = (box[1] + box[3] / 2.0 - transform.top) / transform.scale;

    const int left = std::clamp(static_cast<int>(std::floor(x1)), 0, source_size.width);
    const int top = std::clamp(static_cast<int>(std::floor(y1)), 0, source_size.height);
    const int right = std::clamp(static_cast<int>(std::ceil(x2)), 0, source_size.width);
    const int bottom = std::clamp(static_cast<int>(std::ceil(y2)), 0, source_size.height);
    if (right <= left || bottom <= top) {
        return {};
    }
    return {left, top, right - left, bottom - top};
}

void validate_outputs(std::span<const cv::Mat> outputs)
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
        throw std::runtime_error(
            "Detection rows are too short for box, scores and mask coefficients");
    }
}

// Maps the source-space box into the prototype grid, mixes the coefficients
// with only that prototype region and upsamples it straight to the box size.
// Reconstructing the full-frame mask per detection (prototype grid -> network
// input -> source frame) is much slower when only the box region is kept.
[[nodiscard]] cv::Mat reconstruct_box_mask(std::span<const float> coefficients,
                                           const cv::Mat& prototypes,
                                           const LetterboxResult& transform,
                                           const cv::Rect& box, const Yolov8Config& config)
{
    const int channels = prototypes.size[1];
    const int proto_height = prototypes.size[2];
    const int proto_width = prototypes.size[3];

    const double fx = static_cast<double>(proto_width) / config.input_width;
    const double fy = static_cast<double>(proto_height) / config.input_height;
    const auto to_proto_x = [&](int source_x) {
        return (transform.left + source_x * transform.scale) * fx;
    };
    const auto to_proto_y = [&](int source_y) {
        return (transform.top + source_y * transform.scale) * fy;
    };

    const int x0 = std::clamp(
        static_cast<int>(std::floor(to_proto_x(box.x))), 0, proto_width - 1);
    const int y0 = std::clamp(
        static_cast<int>(std::floor(to_proto_y(box.y))), 0, proto_height - 1);
    const int x1 = std::clamp(
        static_cast<int>(std::ceil(to_proto_x(box.x + box.width))), x0 + 1, proto_width);
    const int y1 = std::clamp(
        static_cast<int>(std::ceil(to_proto_y(box.y + box.height))), y0 + 1, proto_height);
    const int region_height = y1 - y0;
    const int region_width = x1 - x0;

    const std::array<cv::Range, 4> region{
        cv::Range(0, 1), cv::Range::all(), cv::Range(y0, y1), cv::Range(x0, x1)};
    cv::Mat region_prototypes =
        prototypes(region.data()).clone().reshape(1, channels);

    const cv::Mat coefficient_matrix(
        1, channels, CV_32F, const_cast<float*>(coefficients.data()));
    cv::Mat logits = coefficient_matrix * region_prototypes;
    logits = logits.reshape(1, region_height);

    cv::Mat negative_exp;
    cv::exp(-logits, negative_exp);
    const cv::Mat probabilities = 1.0 / (1.0 + negative_exp);

    cv::Mat box_probabilities;
    cv::resize(probabilities, box_probabilities, box.size(), 0.0, 0.0, cv::INTER_LINEAR);
    return box_probabilities > config.mask_threshold;
}

} // namespace

Yolov8Segmenter::Yolov8Segmenter(const std::filesystem::path& model_path,
                                 Yolov8Config config)
    : config_(config),
      engine_(make_engine(config.engine, model_path))
{
    if (config_.input_width <= 0 || config_.input_height <= 0) {
        throw std::invalid_argument("YOLOv8 input dimensions must be positive");
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
    const std::span<const cv::Mat> outputs = engine_->forward(blob);
    const auto inference_end = Clock::now();

    const auto postprocess_start = inference_end;
    validate_outputs(outputs);
    const cv::Mat& detection_tensor = outputs[0];
    const cv::Mat& prototypes = outputs[1];
    const int class_count = detection_tensor.size[1] - 4 - prototypes.size[1];
    const int anchor_count = detection_tensor.size[2];

    // The tensor is feature-major: feature f is a contiguous row of
    // anchor_count values.
    const auto feature_row = [&, tensor = detection_tensor.ptr<float>()](int feature) {
        return tensor + static_cast<std::ptrdiff_t>(feature) * anchor_count;
    };
    const auto wrap_row = [&](const float* row) {
        return cv::Mat(1, anchor_count, CV_32F, const_cast<float*>(row));
    };

    // Element-wise maximum over the class-score rows gives the best score per
    // anchor in a few cache-friendly SIMD passes, avoiding a transpose of the
    // whole tensor.
    cv::Mat best_scores_mat = wrap_row(feature_row(4)).clone();
    for (int class_id = 1; class_id < class_count; ++class_id) {
        cv::max(best_scores_mat, wrap_row(feature_row(4 + class_id)), best_scores_mat);
    }
    const std::span<const float> best_scores{best_scores_mat.ptr<float>(),
                                             static_cast<std::size_t>(anchor_count)};

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<int> anchors;
    for (int anchor = 0; anchor < anchor_count; ++anchor) {
        const float best_score = best_scores[static_cast<std::size_t>(anchor)];
        if (!std::isfinite(best_score) || best_score < config_.confidence_threshold) {
            continue;
        }

        const cv::Rect box = map_box_to_source(
            {feature_row(0)[anchor], feature_row(1)[anchor], feature_row(2)[anchor],
             feature_row(3)[anchor]},
            transformed, image.size());
        if (box.empty()) {
            continue;
        }

        const auto matches_best = [&](int class_id) {
            return feature_row(4 + class_id)[anchor] == best_score;
        };
        const auto class_id =
            *std::ranges::find_if(std::views::iota(0, class_count), matches_best);
        class_ids.push_back(class_id);
        confidences.push_back(best_score);
        boxes.push_back(box);
        anchors.push_back(anchor);
    }

    std::vector<int> kept;
    cv::dnn::NMSBoxes(boxes, confidences, config_.confidence_threshold,
                      config_.nms_threshold, kept);

    // Mask coefficients are gathered only for the detections that survived NMS.
    const int coefficient_count = prototypes.size[1];
    std::vector<float> coefficients(static_cast<std::size_t>(coefficient_count));
    prediction.detections.reserve(kept.size());
    for (const int index : kept) {
        for (int channel = 0; channel < coefficient_count; ++channel) {
            coefficients[static_cast<std::size_t>(channel)] =
                feature_row(4 + class_count + channel)[anchors[index]];
        }
        prediction.detections.push_back(Detection{
            .class_id = class_ids[index],
            .confidence = confidences[index],
            .box = boxes[index],
            .box_mask = reconstruct_box_mask(coefficients, prototypes, transformed,
                                             boxes[index], config_),
        });
    }
    const auto postprocess_end = Clock::now();

    prediction.timing = StageTiming{
        .preprocess_ms = elapsed_ms(preprocess_start, preprocess_end),
        .inference_ms = elapsed_ms(inference_start, inference_end),
        .postprocess_ms = elapsed_ms(postprocess_start, postprocess_end),
        .total_ms = elapsed_ms(total_start, postprocess_end),
    };
    return prediction;
}

} // namespace yolo_seg
