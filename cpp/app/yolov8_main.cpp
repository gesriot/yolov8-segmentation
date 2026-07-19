#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
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

void print_usage(const char* executable)
{
    std::cerr << "Usage: " << executable
              << " --model MODEL.onnx --input IMAGE --output RESULT.png --json RESULT.json\n"
                 "          [--classes CLASSES.txt] [--conf 0.25] [--nms 0.45] [--mask-threshold 0.5]\n"
                 "          [--warmup 1] [--iterations 1]\n";
}

Options parse_options(int argc, char** argv)
{
    Options options;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::runtime_error("Missing value for argument: " + argument);
        }

        const std::string value = argv[++index];
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
        else if (argument == "--conf") {
            options.config.confidence_threshold = std::stof(value);
        }
        else if (argument == "--nms") {
            options.config.nms_threshold = std::stof(value);
        }
        else if (argument == "--mask-threshold") {
            options.config.mask_threshold = std::stof(value);
        }
        else if (argument == "--warmup") {
            options.warmup = std::stoi(value);
        }
        else if (argument == "--iterations") {
            options.iterations = std::stoi(value);
        }
        else {
            throw std::runtime_error("Unknown argument: " + argument);
        }
    }

    if (options.model.empty() || options.input.empty() || options.output.empty()
        || options.json.empty()) {
        throw std::runtime_error("All --model, --input, --output and --json arguments are required");
    }
    if (options.warmup < 0 || options.iterations < 1) {
        throw std::runtime_error("--warmup must be >= 0 and --iterations must be >= 1");
    }
    return options;
}

std::vector<std::string> load_class_names(const std::filesystem::path& path)
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
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            names.push_back(line);
        }
    }
    return names;
}

std::string class_name_for(int class_id, const std::vector<std::string>& class_names)
{
    if (class_id >= 0 && static_cast<std::size_t>(class_id) < class_names.size()) {
        return class_names[static_cast<std::size_t>(class_id)];
    }
    return "";
}

std::string json_escape(const std::string& value)
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

cv::Scalar color_for_class(int class_id)
{
    return {
        static_cast<double>((37 * class_id + 53) % 256),
        static_cast<double>((17 * class_id + 149) % 256),
        static_cast<double>((29 * class_id + 97) % 256)
    };
}

cv::Mat render_result(const cv::Mat& source,
                      const std::vector<yolo_seg::Detection>& detections,
                      const std::vector<std::string>& class_names)
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
            label = "class-" + std::to_string(detection.class_id);
        }
        label += ":" + cv::format("%.4f", detection.confidence);
        const int top = std::max(detection.box.y, 18);
        cv::putText(rendered, label, cv::Point(detection.box.x, top),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);
    }

    cv::addWeighted(rendered, 0.5, mask_overlay, 0.5, 0.0, rendered);
    return rendered;
}

void write_json(const Options& options, const cv::Mat& image,
                const std::vector<yolo_seg::Detection>& detections,
                const std::vector<std::string>& class_names)
{
    std::ofstream stream(options.json);
    if (!stream) {
        throw std::runtime_error("Cannot open JSON output: " + options.json.string());
    }

    stream << std::fixed << std::setprecision(6);
    stream << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"implementation\": \"yolov8-cpp\",\n"
           << "  \"opencv_version\": \"" << CV_VERSION << "\",\n"
           << "  \"model\": \"" << json_escape(options.model.filename().string()) << "\",\n"
           << "  \"input\": \"" << json_escape(options.input.filename().string()) << "\",\n"
           << "  \"image_size\": {\"width\": " << image.cols
           << ", \"height\": " << image.rows << "},\n"
           << "  \"detections\": [\n";

    for (std::size_t index = 0; index < detections.size(); ++index) {
        const auto& detection = detections[index];
        const std::string class_name = class_name_for(detection.class_id, class_names);
        const int mask_pixels =
            detection.box_mask.empty() ? 0 : cv::countNonZero(detection.box_mask);

        stream << "    {\"class_id\": " << detection.class_id
               << ", \"class_name\": \"" << json_escape(class_name)
               << "\", \"confidence\": " << detection.confidence
               << ", \"box\": {\"x\": " << detection.box.x
               << ", \"y\": " << detection.box.y
               << ", \"width\": " << detection.box.width
               << ", \"height\": " << detection.box.height << "}"
               << ", \"mask_pixels\": " << mask_pixels << "}";
        if (index + 1 != detections.size()) {
            stream << ',';
        }
        stream << '\n';
    }
    stream << "  ]\n}\n";
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
        const Options options = parse_options(argc, argv);
        if (!std::filesystem::is_regular_file(options.model)) {
            throw std::runtime_error("Model does not exist: " + options.model.string());
        }
        if (!std::filesystem::is_regular_file(options.input)) {
            throw std::runtime_error("Input image does not exist: " + options.input.string());
        }

        const std::vector<std::string> class_names = load_class_names(options.classes);

        cv::Mat image = cv::imread(options.input.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("OpenCV cannot decode input image: " + options.input.string());
        }

        yolo_seg::Yolov8Segmenter segmenter(options.model, options.config);

        for (int iteration = 0; iteration < options.warmup; ++iteration) {
            segmenter.predict(image);
        }

        yolo_seg::Prediction prediction;
        for (int iteration = 0; iteration < options.iterations; ++iteration) {
            prediction = segmenter.predict(image);
        }

        ensure_parent_directory(options.output);
        ensure_parent_directory(options.json);
        const cv::Mat rendered = render_result(image, prediction.detections, class_names);
        if (!cv::imwrite(options.output.string(), rendered)) {
            throw std::runtime_error("OpenCV cannot write output image: " + options.output.string());
        }
        write_json(options, image, prediction.detections, class_names);

        std::cout << "OpenCV " << CV_VERSION << '\n'
                  << "Detections: " << prediction.detections.size() << '\n'
                  << std::fixed << std::setprecision(2)
                  << "Timing (last iteration): preprocess "
                  << prediction.timing.preprocess_ms << " ms, inference "
                  << prediction.timing.inference_ms << " ms, postprocess "
                  << prediction.timing.postprocess_ms << " ms, total "
                  << prediction.timing.total_ms << " ms\n"
                  << "Image: " << options.output << '\n'
                  << "JSON: " << options.json << '\n';
        return 0;
    }
    catch (const cv::Exception& error) {
        std::cerr << "OpenCV error: " << error.what() << '\n';
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
    }

    print_usage(argv[0]);
    return 1;
}
