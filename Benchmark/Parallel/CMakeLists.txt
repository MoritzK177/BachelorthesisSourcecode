cmake_minimum_required(VERSION 3.16)
project(fastmarchingParallelopenmp1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE "Release")


add_executable(fastmarchingParallelopenmp1 main.cpp settings.h HeapAndStructVector.cpp HeapAndStructVector.h)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fastmarchingParallelopenmp1 PUBLIC OpenMP::OpenMP_CXX)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")
