#include "yolo_seg/inference.hpp"

#include <format>
#include <stdexcept>

#include "engines.hpp"

namespace yolo_seg {

std::unique_ptr<InferenceEngine> make_engine(Engine engine,
                                             const std::filesystem::path& model_path)
{
    switch (engine) {
    case Engine::onnxruntime:
#ifdef YOLO_SEG_WITH_ONNXRUNTIME
        return detail::make_onnxruntime_engine(model_path);
#else
        throw std::runtime_error("This build does not include ONNX Runtime");
#endif
    case Engine::opencv_dnn:
        return detail::make_opencv_engine(model_path);
    }
    throw std::invalid_argument(
        std::format("Unknown inference engine: {}", static_cast<int>(engine)));
}

bool onnxruntime_available() noexcept
{
#ifdef YOLO_SEG_WITH_ONNXRUNTIME
    return true;
#else
    return false;
#endif
}

} // namespace yolo_seg
