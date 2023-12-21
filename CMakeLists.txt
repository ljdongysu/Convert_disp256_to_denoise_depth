cmake_minimum_required(VERSION 3.0.0)
set(PROJECT_NAME denoise)
project(${PROJECT_NAME})
set(CMAKE_CXX_STANDARD 14)
add_definitions(-std=c++11)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)
# opencv
find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})