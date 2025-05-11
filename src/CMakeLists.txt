cmake_minimum_required(VERSION 3.10)
project(test)

set(CMAKE_CXX_STANDARD 17)

include_directories(src)

add_executable(test part2moban.cpp)
