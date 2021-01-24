#pragma once

#include <device_launch_parameters.h>

/*
 * Trying out different implementation of prefix-scan
 * https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
 */

__global__
void scan_1(int* input, int num_elements);

__global__
void scan_2(int* input, int num_elements);

__global__
void scan_3(int* input, int num_elements);
