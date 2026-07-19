#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <print>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "yolo_seg/yolov8_segmentation.hpp"

namespace {

struct Options {
    std::filesystem::path model;
    std::filesystem::path input;
    std::filesystem::path output;
    std::filesystem::path json;
    std::filesystem::path classes;
    yolo_seg::Yolov8Config config;
    int warmup = 1;
    int iterations = 1;
};

void print_usage(std::string_view executable)
{
    std::println(stderr,
                 "Usage: {} --model MODEL.onnx --input IMAGE --output RESULT.png --json RESULT.json\n"
                 "          [--classes CLASSES.txt] [--engine ort|opencv] [--conf 0.25] [--nms 0.45]\n"
                 "          [--mask-threshold 0.5] [--warmup 1] [--iterations 1]",
                 executable);
}

template <typename Value>
[[nodiscard]] Value parse_number(std::string_view argument, std::string_view text)
{
    Value value{};
    const auto [rest, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || rest != text.data() + text.size()) {
        throw std::runtime_error(std::format("Invalid value for {}: {}", argument, text));
    }
    return value;
}

[[nodiscard]] Options parse_options(std::span<char*> arguments)
{
    Options options;

    for (std::size_t index = 1; index < arguments.size(); ++index) {
        const std::string_view argument = arguments[index];
        if (argument == "--help" || argument == "-h") {
            print_usage(arguments[0]);
            std::exit(EXIT_SUCCESS);
        }
        if (index + 1 >= arguments.size()) {
            throw std::runtime_error(std::format("Missing value for argument: {}", argument));
        }

        const std::string_view value = arguments[++index];
        if (argument == "--model") {
            options.model = value;
        }
        else if (argument == "--input") {
            options.input = value;
        }
        else if (argument == "--output") {
            options.output = value;
        }
        else if (argument == "--json") {
            options.json = value;
        }
        else if (argument == "--classes") {
            options.classes = value;
        }
        else if (argument == "--engine") {
            if (value == "ort" || value == "onnxruntime") {
                options.config.engine = yolo_seg::Engine::onnxruntime;
            }
            else if (value == "opencv" || value == "opencv-dnn") {
                options.config.engine = yolo_seg::Engine::opencv_dnn;
            }
            else {
                throw std::runtime_error(
                    std::format("Unknown engine (expected ort or opencv): {}", value));
            }
        }
        else if (argument == "--conf") {
            options.config.confidence_threshold = parse_number<float>(argument, value);
        }
        else if (argument == "--nms") {
            options.config.nms_threshold = parse_number<float>(argument, value);
        }
        else if (argument == "--mask-threshold") {
            options.config.mask_threshold = parse_number<float>(argument, value);
        }
        else if (argument == "--warmup") {
            options.warmup = parse_number<int>(argument, value);
        }
        else if (argument == "--iterations") {
            options.iterations = parse_number<int>(argument, value);
        }
        else {
            throw std::runtime_error(std::format("Unknown argument: {}", argument));
        }
    }

    if (options.model.empty() || options.input.empty() || options.output.empty()
        || options.json.empty()) {
        throw std::runtime_error(
            "All --model, --input, --output and --json arguments are required");
    }
    if (options.warmup < 0 || options.iterations < 1) {
        throw std::runtime_error("--warmup must be >= 0 and --iterations must be >= 1");
    }
    return options;
}

[[nodiscard]] std::vector<std::string> load_class_names(const std::filesystem::path& path)
{
    std::vector<std::string> names;
    if (path.empty()) {
        return names;
    }

    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Cannot open class list: " + path.string());
    }
    std::string line;
    while (std::getline(stream, line)) {
        if (line.ends_with('\r')) {
            line.pop_back();
        }
        if (!line.empty()) {
            names.push_back(line);
        }
    }
    return names;
}

[[nodiscard]] std::string class_name_for(int class_id,
                                         std::span<const std::string> class_names)
{
    if (class_id >= 0 && static_cast<std::size_t>(class_id) < class_names.size()) {
        return class_names[static_cast<std::size_t>(class_id)];
    }
    return {};
}

[[nodiscard]] std::string json_escape(std::string_view value)
{
    std::string escaped;
    escaped.reserve(value.size());
    for (const char character : value) {
        switch (character) {
        case '\\': escaped += "\\\\"; break;
        case '"': escaped += "\\\""; break;
        case '\n': escaped += "\\n"; break;
        case '\r': escaped += "\\r"; break;
        case '\t': escaped += "\\t"; break;
        default: escaped += character; break;
        }
    }
    return escaped;
}

[[nodiscard]] cv::Scalar color_for_class(int class_id) noexcept
{
    return {
        static_cast<double>((37 * class_id + 53) % 256),
        static_cast<double>((17 * class_id + 149) % 256),
        static_cast<double>((29 * class_id + 97) % 256)
    };
}

[[nodiscard]] cv::Mat render_result(const cv::Mat& source,
                                    std::span<const yolo_seg::Detection> detections,
                                    std::span<const std::string> class_names)
{
    cv::Mat rendered = source.clone();
    cv::Mat mask_overlay = source.clone();

    for (const auto& detection : detections) {
        const cv::Scalar color = color_for_class(detection.class_id);
        cv::rectangle(rendered, detection.box, color, 2, cv::LINE_8);

        if (!detection.box_mask.empty()
            && detection.box_mask.size() == detection.box.size()) {
            mask_overlay(detection.box).setTo(color, detection.box_mask);
        }

        std::string label = class_name_for(detection.class_id, class_names);
        if (label.empty()) {
            label = std::format("class-{}", detection.class_id);
        }
        label += std::format(":{:.4f}", detection.confidence);
        const int top = std::max(detection.box.y, 18);
        cv::putText(rendered, label, cv::Point(detection.box.x, top),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);
    }

    cv::addWeighted(rendered, 0.5, mask_overlay, 0.5, 0.0, rendered);
    return rendered;
}

void write_json(const Options& options, const cv::Mat& image,
                std::span<const yolo_seg::Detection> detections,
                std::span<const std::string> class_names)
{
    std::ofstream stream(options.json);
    if (!stream) {
        throw std::runtime_error("Cannot open JSON output: " + options.json.string());
    }

    std::print(stream,
               "{{\n"
               "  \"schema_version\": 1,\n"
               "  \"implementation\": \"yolov8-cpp\",\n"
               "  \"engine\": \"{}\",\n"
               "  \"opencv_version\": \"{}\",\n"
               "  \"model\": \"{}\",\n"
               "  \"input\": \"{}\",\n"
               "  \"image_size\": {{\"width\": {}, \"height\": {}}},\n"
               "  \"detections\": [\n",
               yolo_seg::to_string(options.config.engine), CV_VERSION,
               json_escape(options.model.filename().string()),
               json_escape(options.input.filename().string()), image.cols, image.rows);

    for (std::size_t index = 0; index < detections.size(); ++index) {
        const auto& detection = detections[index];
        const std::string class_name = class_name_for(detection.class_id, class_names);
        const int mask_pixels =
            detection.box_mask.empty() ? 0 : cv::countNonZero(detection.box_mask);

        std::print(stream,
                   "    {{\"class_id\": {}, \"class_name\": \"{}\", \"confidence\": {:.6f}, "
                   "\"box\": {{\"x\": {}, \"y\": {}, \"width\": {}, \"height\": {}}}, "
                   "\"mask_pixels\": {}}}{}\n",
                   detection.class_id, json_escape(class_name), detection.confidence,
                   detection.box.x, detection.box.y, detection.box.width,
                   detection.box.height, mask_pixels,
                   index + 1 == detections.size() ? "" : ",");
    }
    std::print(stream, "  ]\n}}\n");
}

void ensure_parent_directory(const std::filesystem::path& path)
{
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Options options = parse_options({argv, static_cast<std::size_t>(argc)});
        if (!std::filesystem::is_regular_file(options.model)) {
            throw std::runtime_error("Model does not exist: " + options.model.string());
        }
        if (!std::filesystem::is_regular_file(options.input)) {
            throw std::runtime_error("Input image does not exist: " + options.input.string());
        }

        const std::vector<std::string> class_names = load_class_names(options.classes);

        const cv::Mat image = cv::imread(options.input.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error(
                "OpenCV cannot decode input image: " + options.input.string());
        }

        yolo_seg::Yolov8Segmenter segmenter(options.model, options.config);

        for (int iteration = 0; iteration < options.warmup; ++iteration) {
            std::ignore = segmenter.predict(image);
        }

        yolo_seg::Prediction prediction;
        std::vector<double> inference_samples;
        inference_samples.reserve(static_cast<std::size_t>(options.iterations));
        for (int iteration = 0; iteration < options.iterations; ++iteration) {
            prediction = segmenter.predict(image);
            inference_samples.push_back(prediction.timing.inference_ms);
        }

        ensure_parent_directory(options.output);
        ensure_parent_directory(options.json);
        const cv::Mat rendered = render_result(image, prediction.detections, class_names);
        if (!cv::imwrite(options.output.string(), rendered)) {
            throw std::runtime_error(
                "OpenCV cannot write output image: " + options.output.string());
        }
        write_json(options, image, prediction.detections, class_names);

        std::println("OpenCV {}, engine {}", CV_VERSION,
                     yolo_seg::to_string(options.config.engine));
        std::println("Detections: {}", prediction.detections.size());
        std::println("Timing (last iteration): preprocess {:.2f} ms, inference {:.2f} ms, "
                     "postprocess {:.2f} ms, total {:.2f} ms",
                     prediction.timing.preprocess_ms, prediction.timing.inference_ms,
                     prediction.timing.postprocess_ms, prediction.timing.total_ms);
        if (options.iterations > 1) {
            std::ranges::sort(inference_samples);
            const auto median =
                inference_samples[inference_samples.size() / 2];
            std::println("Inference over {} iterations: min {:.2f} ms, median {:.2f} ms",
                         options.iterations, inference_samples.front(), median);
        }
        std::println("Image: {}", options.output.string());
        std::println("JSON: {}", options.json.string());
        return EXIT_SUCCESS;
    }
    catch (const cv::Exception& error) {
        std::println(stderr, "OpenCV error: {}", error.what());
    }
    catch (const std::exception& error) {
        std::println(stderr, "Error: {}", error.what());
    }

    print_usage(argv[0]);
    return EXIT_FAILURE;
}
