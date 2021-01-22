#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "src/math/vector_type.h"
#include "src/data/chunklist.h"
#include "src/data/mesh_aabb.h"
#include "src/data/kernel/reduction.h"

namespace CuKee
{
  __host__ __device__ __inline__
  void cu_clip_line()
  {

  }

  __host__ __device__ __inline__
  void cu_clip_triangle()
  {

  }

  /*
   * Transform the "un-chunkified" chunklist
   */
  __global__
  void cu_create_chunk_from_nodes(Device::ChunkList& chunklist)
  {

  }

  __global__
  void sort_prim(Device::ChunkList& activelist,
                    unsigned int num_prim,
                    int* tags)
  {
    /* Variables shared across the thread block */
    __shared__ int split_axis;
    __shared__ float split_pos;
    __shared__ unsigned int starting_index;
    __shared__ unsigned int n_prim;
    __shared__ unsigned int node_index;
    __shared__ unsigned int chunk_index;

    int thread = threadIdx.x;
    if (thread == 0)
    {
      chunk_index = blockIdx.x;
      node_index = activelist.m_nodelist.
      split_axis = activelist.m_nodelist.m_split_data.m_split_axis[0];
      split_pos = activelist.m_nodelist.m_split_data.m_split_position[0];
    }
    __syncthreads();
  }
}
