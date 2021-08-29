cmake_minimum_required(VERSION 3.17)

###############################################################################
# Project definition

project(CUDA-kdtree
        VERSION 0.0.1
        DESCRIPTION "GPU-acclerated KD-tree using CUDA"
        LANGUAGES CUDA CXX)

enable_testing()

#############################################################################
# CMAKE configuration

set(CMAKE_WARN_DEPRECATED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

###############################################################################
# OpenGL dependencies

find_package(OpenGL REQUIRED COMPONENTS OpenGL)

if(NOT APPLE)
    find_package(GLEW REQUIRED)
endif()

find_package(GLFW3 REQUIRED)
find_package(GLM REQUIRED)

###############################################################################
# CUDA dependencies

find_package(CUDAToolkit REQUIRED)
find_library(CUDA_NVTX_LIBRARY
        NAMES nvToolsExt nvTools
        PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

set(PROFILE_MODE ON)
if(NOT CUDA_NVTX_LIBRARY)
    message(STATUS "NVIDIA Extension Library not found.")
    set(PROFILE_MODE OFF)
endif()

###############################################################################
# CUDA CMake variables
set(CMAKE_CUDA_FLAGS "-Wall")

###############################################################################
# General dependencies

find_package(ASSIMP REQUIRED)
find_package(Boost REQUIRED)

###############################################################################
# Progress to other sources

add_subdirectory(ext)
add_subdirectory(src)
add_subdirectory(tests)