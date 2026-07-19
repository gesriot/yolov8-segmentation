#pragma once

#include <filesystem>
#include <memory>
#include <span>
#include <string_view>

#include <opencv2/core.hpp>

namespace yolo_seg {

enum class Engine {
    onnxruntime,
    opencv_dnn,
};

[[nodiscard]] constexpr std::string_view to_string(Engine engine) noexcept
{
    switch (engine) {
    case Engine::onnxruntime: return "onnxruntime";
    case Engine::opencv_dnn: return "opencv-dnn";
    }
    return "unknown";
}

// Runs a loaded network on preprocessed input. Implementations are not
// thread-safe; use one engine per thread.
class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;

    // Runs the network on an NCHW CV_32F blob. The returned tensors stay
    // valid until the next forward() call or engine destruction.
    [[nodiscard]] virtual std::span<const cv::Mat> forward(const cv::Mat& blob) = 0;
};

// Loads the model with the requested engine. Throws std::runtime_error when
// the engine is unavailable in this build or the model cannot be loaded.
[[nodiscard]] std::unique_ptr<InferenceEngine> make_engine(
    Engine engine, const std::filesystem::path& model_path);

[[nodiscard]] bool onnxruntime_available() noexcept;

} // namespace yolo_seg
