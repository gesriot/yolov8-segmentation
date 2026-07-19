#pragma once

#include <filesystem>
#include <memory>

#include "yolo_seg/inference.hpp"

namespace yolo_seg::detail {

[[nodiscard]] std::unique_ptr<InferenceEngine> make_opencv_engine(
    const std::filesystem::path& model_path);

#ifdef YOLO_SEG_WITH_ONNXRUNTIME
[[nodiscard]] std::unique_ptr<InferenceEngine> make_onnxruntime_engine(
    const std::filesystem::path& model_path);
#endif

} // namespace yolo_seg::detail
