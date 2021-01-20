// GL Dependencies
#include <glm/glm.hpp>

// CUDA Dependencies
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#include "src/kdtreeGPU/reduction.h"
#include "src/kdtreeGPU/sharedmemory.h"
#include "src/utils/cuda_common.h"



template <typename T, class Op>
__global__
void cu_segmented_reduce_1(T *val,
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
  int thread = threadIdx.x;
  int index = blockIdx.x * blockDim.x + thread;

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
void cu_chunk_reduce(T* val,
                     int* starting_indices,
                     int* chunk_len,
                     T* output)
{
  T* s_arr = SharedMemory<T>();
  unsigned int thread = threadIdx.x;
  unsigned int chunk_index = blockIdx.x;
  unsigned int starting_index = starting_indices[chunk_index];
  unsigned int len = chunk_len[chunk_index];

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
  cu_chunk_reduce<T, Op, chunk_size> <<<>>>();
}
