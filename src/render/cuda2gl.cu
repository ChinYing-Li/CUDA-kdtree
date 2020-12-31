/*
    Here we aim to send data from CUDA to OpenGL.
    Reference: cuda-samples/2_Graphics/simple_GL
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// CUDA Dependencies
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>         // Helper functions for CUDA error check
#include <helper_cuda_gl.h>      // Helper functions for CUDA/GL interop
#include <helper_functions.h>    // Includes cuda.h and cuda_runtime_api.h
#include <timer.h>

// OpenGL Dependencies
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <vector_types.h>

