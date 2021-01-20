#pragma once

#include <cuda_runtime.h>

#include <stdlib.h>
#include <iostream>

// TODO: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api

void checkCudaError(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    std::cerr << "CUDA Error: "
              << msg << "\n"
              << cudaGetErrorString(err)
              << std::endl;
    exit(1);
  }
}
