// GL Dependencies
#include <glm/glm.hpp>

// CUDA Dependencies
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "src/data/kernel/reduction.h"
#include "src/utils/sharedmemory.h"
#include "src/utils/cuda_common.h"

/*
 * Various kernels for reduction.
 * See https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 */
template <unsigned int block_size, typename T, class Op>
__inline__ __device__
T block_reduce_1(T* input)
{
  extern __shared__ T shared_data[block_size];
  unsigned int thread = threadIdx.x;
  unsigned int index = blockIdx.x * blockDim.x + thread;
  shared_data[thread] = input[index];
  __syncthreads();

  // I think blockDim.x should be equal to block_size
  for(unsigned int s = 1; s < blockDim.x; ++s)
  {
    if(thread % (2*s) == 0)
    {

      shared_data[thread] = Op::op(shared_data[thread], shared_data[thread + s]);
    }
    __syncthreads();
  }
  if (thread == 0)
  {
    return shared_data[thread];
  }
}

/*
 * Interleaved addressing, which leads to bank conflict
 */
template <unsigned int block_size, typename T, class Op>
__inline__ __device__
T block_reduce_2(T* input)
{
  extern __shared__ T shared_data[block_size];
  unsigned int thread = threadIdx.x;
  unsigned int index = blockIdx.x * blockDim.x + thread;
  shared_data[thread] = input[index];
  __syncthreads();

  for (unsigned int s = 1; s < block_size; ++s)
  {
    int current = 2 * thread * s;
    if (current + s < block_size)
    {
      shared_data[current] = Op::op(shared_data[thread], shared_data[thread + s]);
    }
    __syncthreads();
  }
  if (thread == 0)
  {
    return shared_data[thread];
  }
}

/*
 * Sequential addressing
 */
template <unsigned int block_size, typename T, class Op>
__inline__ __device__
T block_reduce_3(T* input)
{
  extern __shared__ T shared_data[block_size];
  unsigned int thread = threadIdx.x;
  unsigned int index = blockIdx.x * blockDim.x + thread;
  shared_data[thread] = input[index];
  __syncthreads();

  // Half of the threads are idle on the first loop iteration
  for (unsigned int s = block_size / 2; s > 0; s >>= 1)
  {
   if (thread < s)
   {
     shared_data[thread] = Op::op(shared_data[thread], shared_data[thread + s]);
   }
    __syncthreads();
  }
  if (thread == 0)
  {
    return shared_data[thread];
  }
}

/*
 * Sequential addressing and first reduction outside the loop
 * Halve block_size (== blockDim.x) in the template
 * TODO: why do we even need this first template parameter?
 */
template <unsigned int halved_block_size, typename T, class Op>
__inline__ __device__
T block_reduce_4(T* input)
{
  extern __shared__ T shared_data[halved_block_size];
  unsigned int thread = threadIdx.x;
  unsigned int index = blockIdx.x * blockDim.x * 2 + thread;
  shared_data[thread] = input[index] + input[index + halved_block_size];
  __syncthreads();

  // Half of the threads are idle on the first loop iteration
  for (unsigned int s = block_size / 2; s > 0; s >>= 1)
  {
   if (thread < s)
   {
     shared_data[thread] = Op::op(shared_data[thread], shared_data[thread + s]);
   }
    __syncthreads();
  }

  if (thread == 0)
  {
    return shared_data[thread];
  }
}

/*
 * When s <= 32, only one warp (unit for SIMD) exists.
 * Unroll the last six loop by defining "warp_reduce".
 * Use the volatile keyword, according to
 * https://stackoverflow.com/questions/15331009/when-to-use-volatile-with-shared-cuda-memory
 */
template <typename T>
__device__
void warp_reduce(volatile T* shared_data, int thread)
{
  shared_data[thread] += shared_data[thread + 32];
  shared_data[thread] += shared_data[thread + 16];
  shared_data[thread] += shared_data[thread +  8];
  shared_data[thread] += shared_data[thread +  4];
  shared_data[thread] += shared_data[thread +  2];
  shared_data[thread] += shared_data[thread +  1];
}

// TODO: the return type is incorrect
template <unsigned int halved_block_size, typename T, class Op>
__inline__ __device__
T block_reduce_5(T* input)
{
  extern __shared__ T shared_data[halved_block_size];
  unsigned int thread = threadIdx.x;
  unsigned int index = blockIdx.x * blockDim.x * 2 + thread;
  shared_data[thread] = input[index] + input[index + halved_block_size];
  __syncthreads();

  for (unsigned int s = block_size / 2; s > 32; s >>= 1)
  {
   if (thread < s)
   {
     shared_data[thread] = Op::op(shared_data[thread], shared_data[thread + s]);
   }
    __syncthreads();
  }

  if (thread < 32)
  {
    warp_reduce(shared_data, thread);
  }
  __syncthread();

  if (thread == 0)
  {
    return shared_data[thread];
  }
}

template <typename T, class Op>
__global__
void segmented_reduce_1(T *val,
                         unsigned int num_val,
                         int* keys,
                         unsigned int num_keys,
                         int* key_ranges,
                         T* output)
{
  __shared__ int s_key[256];
  __shared__ int s_val[256];
  T* s_arr = SharedMemory<T>();

  int min_key = key_ranges[2 * blockIdx.x];
  int key_diff = key_ranges[2 * blockIdx.x + 1] - min_key;
  unsigned int thread = threadIdx.x;
  unsigned int index = blockIdx.x * blockDim.x + thread;

  s_val[thread] = val[thread];
  s_key[thread] = keys[thread];
  if(thread <= key_diff)
  {
    s_arr[thread] = 0;
  }

  __syncthreads();

  for(int i = 1; i < blockDim.x; i *= 2)
  {
    if(thread % (i * 2) == 0) // Branch divergence!
    {
      int w0 = s_key[thread];
      int w1 = s_key[thread + i];

      if (w0 != w1) // If keys are different
      {
        s_arr[w1 - min_key] += s_val[thread + i];
      }
      else
      {
        s_val[thread] += s_val[thread + i];
      }
    }
    __syncthreads();

    if (thread <= key_diff)
    {
      atomicAdd(&output[min_key + thread], s_arr[thread]);
    }
    __syncthreads();
    if(thread == 0)
    {
      atomicAdd(&output[s_key[0]], s_val[0]);
    }
  }

}

// Key preprocessing for segmented reduction
__global__
void get_keyrange_per_block(int* keys,
                            int* ranges,
                            int n, int block_size)
{
  int thread = threadIdx.x;
  if(thread < block_size)
  {
    ranges[2*thread] = keys[block_size * thread];
    ranges[2*thread + 1] = keys[block_size * (thread+1) - 1];
  }
}

/*
 *
 */
template <typename T, class Op>
void segmented_reduce(T *val,
                      unsigned int num_val,
                      int* keys,
                      unsigned int num_keys,
                      T* output)
{
  unsigned int num_threads = 256;
  unsigned int num_blocks = num_val / num_threads; // TODO: Remove the constraint on num_val
  int* keyranges;
  cudaMalloc((void**) *keyranges, 2 * num_blocks * sizeof (int));
  checkCudaError("cudaMalloc failure");

  dim3 dim_block_ranges(num_blocks, 1, 1);
  dim3 dim_grid_ranges(1, 1, 1);
  dim3 dim_block(num_threads, 1, 1);
  dim3 dim_grid(num_blocks, 1, 1);
  unsigned int sme_size = num_threads * 3 * sizeof (int); // What does this mean?
  cu_segmented_reduce_1<T, Op><<<dim_grid, dim_block, sme_size>>>(val, num_val, keys, num_keys, keyranges, output);
  checkCudaError("Segmented reduction kernel call failed");
  cudaFree(keyranges);
}

/*
 * This is basically a block reduction, since we are placing
 */
template <typename T, class Op, unsigned int chunk_size>
__global__
void chunk_reduce(T* val,
                     int* starting_indices,
                     int* chunk_len,
                     T* output)
{
  T* s_arr = SharedMemory<T>();
  unsigned int thread = threadIdx.x;
  unsigned int chunk_index = blockIdx.x;
  __shared__ unsigned int starting_index;
  __shared__ unsigned int len;

  if (thread == 0)
  {
    starting_index = starting_indices[chunk_index];
    len = chunk_len[chunk_index];
  }
  __syncthreads();

  if (thread < len) // Branching?
  {
    s_arr[thread] = val[starting_index + thread];
  }
  else
  {
    s_arr[thread] = Op::identity();
  }

  __syncthreads();

  T res = cu_reduce<T, Op, chunk_size>(s_arr);

  if (thread == 0)
  {
    output[chunk_index] = res;
  }
}

template <typename T, class Op, unsigned int chunk_size>
void chunk_reduce(T* val, int* starting_indices,
                  int* chunk_len, unsigned int num_chunks,
                  T* output)
{
  dim3 block_dim = dim3(chunk_size, 1, 1);
  dim3 grid_dim = dim3(num_chunks, 1, 1);
  int size = chunk_size * sizeof (T);
  chunk_reduce<T, Op, chunk_size> <<<block_dim, grid_dim>>>(val, starting_indices, chunk_len, output);
}
