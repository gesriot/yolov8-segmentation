find_package(
    OpenCV REQUIRED
    COMPONENTS core imgproc imgcodecs dnn
)

if(OpenCV_VERSION VERSION_LESS 4.7)
    message(FATAL_ERROR "OpenCV 4.7 or newer is required; found ${OpenCV_VERSION}")
endif()

set(YOLO_SEG_ORT_VERSION "1.27.1" CACHE STRING "Pinned ONNX Runtime version")
set(YOLO_SEG_ORT_ROOT
    "${PROJECT_SOURCE_DIR}/third_party/onnxruntime/${YOLO_SEG_ORT_VERSION}")

set(yolo_seg_sources
    src/inference.cpp
    src/engine_opencv.cpp
    src/yolov8_segmentation.cpp
)

if(EXISTS "${YOLO_SEG_ORT_ROOT}/lib/onnxruntime.dll")
    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(
        onnxruntime
        PROPERTIES
            IMPORTED_LOCATION "${YOLO_SEG_ORT_ROOT}/lib/onnxruntime.dll"
            IMPORTED_IMPLIB "${YOLO_SEG_ORT_ROOT}/lib/onnxruntime.lib"
            INTERFACE_INCLUDE_DIRECTORIES "${YOLO_SEG_ORT_ROOT}/include"
    )
    list(APPEND yolo_seg_sources src/engine_onnxruntime.cpp)
    set(YOLO_SEG_HAVE_ORT TRUE)
else()
    message(WARNING
        "ONNX Runtime was not found in ${YOLO_SEG_ORT_ROOT}; building without it. "
        "Run scripts/get_onnxruntime_windows.ps1 to fetch it.")
    set(YOLO_SEG_HAVE_ORT FALSE)
endif()

add_library(yolo-seg-cpp ${yolo_seg_sources})
target_compile_features(yolo-seg-cpp PUBLIC cxx_std_23)
target_include_directories(
    yolo-seg-cpp
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(yolo-seg-cpp PUBLIC ${OpenCV_LIBS})

if(YOLO_SEG_HAVE_ORT)
    target_compile_definitions(yolo-seg-cpp PRIVATE YOLO_SEG_WITH_ONNXRUNTIME)
    target_link_libraries(yolo-seg-cpp PRIVATE onnxruntime)
endif()

add_executable(yolov8-seg app/yolov8_main.cpp)
target_compile_features(yolov8-seg PRIVATE cxx_std_23)
target_link_libraries(yolov8-seg PRIVATE yolo-seg-cpp)

if(YOLO_SEG_HAVE_ORT)
    add_custom_command(
        TARGET yolov8-seg POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:onnxruntime>" "$<TARGET_FILE_DIR:yolov8-seg>"
    )
endif()

if(BUILD_TESTING)
    set(LOCAL_YOLOV8_IMAGE "${PROJECT_SOURCE_DIR}/shared/testdata/bus.jpg")
    set(LOCAL_COCO_CLASSES "${PROJECT_SOURCE_DIR}/shared/classes/coco80.txt")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/test-output")

    set(yolo_seg_test_engines opencv)
    if(YOLO_SEG_HAVE_ORT)
        list(PREPEND yolo_seg_test_engines ort)
    endif()

    foreach(model_size IN ITEMS n s m)
        set(model_path "${PROJECT_SOURCE_DIR}/models/yolov8/yolov8${model_size}-seg.onnx")
        if(EXISTS "${model_path}" AND EXISTS "${LOCAL_YOLOV8_IMAGE}")
            foreach(engine IN LISTS yolo_seg_test_engines)
                add_test(
                    NAME "yolov8${model_size}-seg-bus-${engine}"
                    COMMAND yolov8-seg
                        --model "${model_path}"
                        --input "${LOCAL_YOLOV8_IMAGE}"
                        --classes "${LOCAL_COCO_CLASSES}"
                        --engine "${engine}"
                        --output "${CMAKE_BINARY_DIR}/test-output/yolov8${model_size}-bus-${engine}.png"
                        --json "${CMAKE_BINARY_DIR}/test-output/yolov8${model_size}-bus-${engine}.json"
                        --warmup 0
                        --iterations 1
                )
                set_tests_properties(
                    "yolov8${model_size}-seg-bus-${engine}"
                    PROPERTIES
                        TIMEOUT 180
                        ENVIRONMENT_MODIFICATION
                            "PATH=path_list_prepend:${OpenCV_INSTALL_PATH}/bin"
                )
            endforeach()
        endif()
    endforeach()
endif()
