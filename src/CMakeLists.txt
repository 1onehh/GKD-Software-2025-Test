cmake_minimum_required(VERSION 3.10)
project(part6_project)

set(CMAKE_CXX_STANDARD 17)

# 头文件目录
include_directories(C:/Users/czx/Desktop/GKD-Software-2025-Test/src/OpenCV-4.9.0-MinGW64-main/include)

# 库文件目录
link_directories(C:/Users/czx/Desktop/GKD-Software-2025-Test/src/OpenCV-4.9.0-MinGW64-main/x64/mingw/bin)

add_executable(part6-2 part6-2.cpp)

# OpenCV 库
target_link_libraries(part6-2
    opencv_world490.dll
)
