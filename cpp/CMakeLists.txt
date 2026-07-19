find_package(
    OpenCV REQUIRED
    COMPONENTS core imgproc imgcodecs dnn
)

if(OpenCV_VERSION VERSION_LESS 4.7)
    message(FATAL_ERROR "OpenCV 4.7 or newer is required; found ${OpenCV_VERSION}")
endif()

add_library(yolo-seg-cpp src/yolov8_segmentation.cpp)
target_compile_features(yolo-seg-cpp PUBLIC cxx_std_23)
target_include_directories(
    yolo-seg-cpp
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(yolo-seg-cpp PUBLIC ${OpenCV_LIBS})

add_executable(yolov8-seg app/yolov8_main.cpp)
target_compile_features(yolov8-seg PRIVATE cxx_std_23)
target_link_libraries(yolov8-seg PRIVATE yolo-seg-cpp)

if(BUILD_TESTING)
    set(LOCAL_YOLOV8_IMAGE "${PROJECT_SOURCE_DIR}/shared/testdata/bus.jpg")
    set(LOCAL_COCO_CLASSES "${PROJECT_SOURCE_DIR}/shared/classes/coco80.txt")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/test-output")

    foreach(model_size IN ITEMS n s m)
        set(model_path "${PROJECT_SOURCE_DIR}/models/yolov8/yolov8${model_size}-seg.onnx")
        if(EXISTS "${model_path}" AND EXISTS "${LOCAL_YOLOV8_IMAGE}")
            add_test(
                NAME "yolov8${model_size}-seg-bus"
                COMMAND yolov8-seg
                    --model "${model_path}"
                    --input "${LOCAL_YOLOV8_IMAGE}"
                    --classes "${LOCAL_COCO_CLASSES}"
                    --output "${CMAKE_BINARY_DIR}/test-output/yolov8${model_size}-bus.png"
                    --json "${CMAKE_BINARY_DIR}/test-output/yolov8${model_size}-bus.json"
                    --warmup 0
                    --iterations 1
            )
            set_tests_properties(
                "yolov8${model_size}-seg-bus"
                PROPERTIES
                    TIMEOUT 180
                    ENVIRONMENT_MODIFICATION
                        "PATH=path_list_prepend:${OpenCV_INSTALL_PATH}/bin"
            )
        endif()
    endforeach()
endif()
