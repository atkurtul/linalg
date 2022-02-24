cmake_minimum_required(VERSION 3.0)

project(linalg)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

add_library(linalg INTERFACE)
target_include_directories(linalg INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
