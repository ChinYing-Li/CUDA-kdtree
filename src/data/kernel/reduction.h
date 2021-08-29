#pragma once

// Thrust Dependencies
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>

/****************************************************************************************
 * Wrapper function for Reduction kernels
 */
template <typename T, class Op>
void segmented_reduce(T *val, unsigned int num_val,
                      int* keys, unsigned int num_keys,
                      T* output);

template <typename T, class Op, unsigned int chunk_size>
void chunk_reduce(T* val, int* starting_indices,
                  int* chunk_len, unsigned int num_chunks,
                  T* output);

// Unsegmented reduction on GPU
template <typename T, class Op, unsigned int block_size>
__device__
T reduce(T* val);

/****************************************************************************************
 * Operators for reduction
 */
template <typename T>
class ReductionOp_Sum
{
  inline static __host__ __device__
  T op(const T& lhs, const T& rhs)
  {
    return lhs + rhs;
  }

  inline static __host__ __device__
  T identity()
  {
    return (T)0;
  }

  inline static __host__ __device__
  int stride()
  {
    return 1;
  }
};

template <typename T>
class ReductionOp_Max
{
  inline static __host__ __device__
  T op(const T& lhs, const T& rhs)
  {
    return max(lhs, rhs);
  }

  inline static __host__ __device__
  T identity()
  { // The identity here should be -infinity ...
    return (T) 0;
  }

  inline static __host__ __device__
  int stride()
  {
    return 1;
  }
};

template <typename T>
class ReductionOp_Min
{
  inline static __host__ __device__
  T op(const T& lhs, const T& rhs)
  {
    return min(lhs, rhs);
  }

  inline static __host__ __device__
  T identity()
  {
    return T(0);
  }

  inline static __host__ __device__
  int stride()
  {
    return 1;
  }
};
