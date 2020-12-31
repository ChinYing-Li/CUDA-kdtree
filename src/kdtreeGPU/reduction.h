#pragma once

// CUDA Dependencies
#include <cuda.h>

// Thrust Dependencies
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>

/*
 * GPU Segmented Reduction, as described in Algorithm 3,
 */
__device__ void segmented_reduction(float *data, float *owner);
