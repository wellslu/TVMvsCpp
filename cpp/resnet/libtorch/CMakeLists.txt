cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(resnet)

set(TORCH_CUDA OFF CACHE BOOL "Disable CUDA support in PyTorch")


find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(resnet main.cpp)
target_link_libraries(resnet "${TORCH_LIBRARIES}")
set_property(TARGET resnet PROPERTY CXX_STANDARD 17)