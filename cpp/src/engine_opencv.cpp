#include "engines.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/dnn.hpp>

namespace yolo_seg::detail {
namespace {

class OpenCvEngine final : public InferenceEngine {
public:
    explicit OpenCvEngine(const std::filesystem::path& model_path)
        : network_(cv::dnn::readNet(model_path.string()))
    {
        network_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        network_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        output_names_ = network_.getUnconnectedOutLayersNames();
    }

    [[nodiscard]] std::span<const cv::Mat> forward(const cv::Mat& blob) override
    {
        network_.setInput(blob);
        network_.forward(outputs_, output_names_);
        return outputs_;
    }

private:
    cv::dnn::Net network_;
    std::vector<std::string> output_names_;
    std::vector<cv::Mat> outputs_;
};

} // namespace

std::unique_ptr<InferenceEngine> make_opencv_engine(const std::filesystem::path& model_path)
{
    return std::make_unique<OpenCvEngine>(model_path);
}

} // namespace yolo_seg::detail
