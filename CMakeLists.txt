cmake_minimum_required(VERSION 3.17)

###############################################################################
# Project definition

project(CUDA-kdtree
        VERSION 0.0.1
        CMAKE_PROJECT_DESCRIPTION "OpenGL Rendering Engine implementing Cluster Shading"
        LANGUAGES CXX C)

enable_testing()
#############################################################################
# CMAKE configuration

set(CMAKE_WARN_DEPRECATED ON)
set(CMAKE_CXX_STANDARD 11 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

###############################################################################
# OpenGL dependencies

find_package(OpenGL REQUIRED COMPONENTS OpenGL)

if(NOT APPLE)
    find_package(GLEW REQUIRED)
endif()

find_package(GLFW3 REQUIRED)
find_package(GLM REQUIRED)
find_package(ASSIMP REQUIRED)

###############################################################################
# CUDA dependencies

find_package(CUDAToolkit REQUIRED)
find_library(NVTX_LIBRARY nvToolsExt PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

set(PROFILE_MODE OFF)
if(NVTX_LIBRARY)
    message(STATUS "NVIDIA Extension Library not found; can't profile with it.")
    set(PROFILE_MODE ON)
endif()

add_executable(CUDA-kdtree src/main.cpp)
