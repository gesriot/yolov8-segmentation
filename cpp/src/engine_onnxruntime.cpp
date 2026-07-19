#include "engines.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace yolo_seg::detail {
namespace {

[[nodiscard]] Ort::SessionOptions make_session_options()
{
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return options;
}

class OnnxRuntimeEngine final : public InferenceEngine {
public:
    explicit OnnxRuntimeEngine(const std::filesystem::path& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "yolo-seg"),
          session_(env_, model_path.c_str(), make_session_options()),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
        const Ort::AllocatorWithDefaultOptions allocator;
        if (session_.GetInputCount() != 1) {
            throw std::runtime_error("YOLOv8 ONNX model must expose one input");
        }
        input_name_ = session_.GetInputNameAllocated(0, allocator).get();

        const std::size_t output_count = session_.GetOutputCount();
        for (std::size_t index = 0; index < output_count; ++index) {
            output_names_.push_back(
                session_.GetOutputNameAllocated(index, allocator).get());
        }
        for (const auto& name : output_names_) {
            output_name_pointers_.push_back(name.c_str());
        }
    }

    [[nodiscard]] std::span<const cv::Mat> forward(const cv::Mat& blob) override
    {
        if (blob.type() != CV_32F || !blob.isContinuous()) {
            throw std::invalid_argument("Inference input must be a continuous CV_32F blob");
        }

        std::vector<std::int64_t> input_shape(blob.size.p, blob.size.p + blob.dims);
        const Ort::Value input = Ort::Value::CreateTensor<float>(
            memory_info_, const_cast<float*>(blob.ptr<float>()), blob.total(),
            input_shape.data(), input_shape.size());

        const std::array<const char*, 1> input_names{input_name_.c_str()};
        outputs_ = session_.Run(Ort::RunOptions{}, input_names.data(), &input, 1,
                                output_name_pointers_.data(), output_name_pointers_.size());

        // Wrap the ONNX Runtime tensors as cv::Mat views; outputs_ keeps the
        // memory alive until the next call.
        views_.clear();
        for (const Ort::Value& output : outputs_) {
            const auto info = output.GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                throw std::runtime_error("YOLOv8 output tensors must be float32");
            }
            const std::vector<std::int64_t> shape = info.GetShape();
            std::vector<int> sizes(shape.begin(), shape.end());
            views_.emplace_back(static_cast<int>(sizes.size()), sizes.data(), CV_32F,
                                const_cast<float*>(output.GetTensorData<float>()));
        }
        return views_;
    }

private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo memory_info_;
    std::string input_name_;
    std::vector<std::string> output_names_;
    std::vector<const char*> output_name_pointers_;
    std::vector<Ort::Value> outputs_;
    std::vector<cv::Mat> views_;
};

} // namespace

std::unique_ptr<InferenceEngine> make_onnxruntime_engine(
    const std::filesystem::path& model_path)
{
    return std::make_unique<OnnxRuntimeEngine>(model_path);
}

} // namespace yolo_seg::detail
