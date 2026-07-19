cmake_minimum_required(VERSION 3.24)

project(
    yolov8_segmentation
    VERSION 1.0.0
    LANGUAGES CXX
)

include(CTest)

add_subdirectory(cpp)
