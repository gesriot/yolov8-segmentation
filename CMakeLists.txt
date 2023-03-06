CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)
project(yolov8-seg)

SET (OpenCV_DIR ../build)  #opencv root

FIND_PACKAGE(OpenCV REQUIRED)
#include_directories("../build/install/include")
ADD_EXECUTABLE(YOLOv8 yolov8_seg.h yolov8_utils.h
			 main.cpp yolov8_seg.cpp yolov8_utils.cpp)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

TARGET_INCLUDE_DIRECTORIES(YOLOv8 PRIVATE "../build/install/include")

TARGET_COMPILE_FEATURES(YOLOv8 PRIVATE cxx_std_14)
TARGET_LINK_LIBRARIES(YOLOv8 ${OpenCV_LIBS})

TARGET_LINK_LIBRARIES(YOLOv8 "../build/install/x64/vc16/lib")