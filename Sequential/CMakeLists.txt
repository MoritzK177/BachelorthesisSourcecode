cmake_minimum_required(VERSION 3.16)
project(fastmarching4)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE "Release")


add_executable(fastmarching4 main.cpp main.cpp settings.h HeapAndStructVector.cpp HeapAndStructVector.h)
#[[ set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O2")]]
