cmake_minimum_required(VERSION 3.10)
project(part6_project)

set(CMAKE_CXX_STANDARD 17)

# OpenCV 安装路径根目录
set(OpenCV_DIR "C:/Users/czx/Desktop/GKD-Software-2025-Test/opencv/build/install")

# 添加头文件路径
include_directories(${OpenCV_DIR}/include)

# 添加库文件目录
link_directories(${OpenCV_DIR}/x64/mingw/lib)

add_executable(part6 part6.cpp)

# 链接所有需要的 OpenCV 库
target_link_libraries(part6
    opencv_core4120
    opencv_imgproc4120
    opencv_highgui4120
    opencv_imgcodecs4120
)
