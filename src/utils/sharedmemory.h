#pragma once

#include <cuda_runtime.h>

template <class T>
struct SharedMemory
{
  __device__ inline operator T*()
  {
    extern __shared__ int sarr___[];
    return (T*) sarr___;
  }

  __device__ inline operator const T*() const
  {
    extern __shared__ int sarr___[];
    return (T*) sarr___;
  }
};
