#pragma once

#include <cuda_runtime.h>

#include <stdlib.h>
#include <iostream>

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
