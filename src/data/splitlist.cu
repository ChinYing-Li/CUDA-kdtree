#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "src/data/chunklist.h"
#include "src/data/splitlist.h"

namespace CuKee
{

namespace Device
{
void Device::SplitData::
clear()
{
  delete [] m_split_offset;
  delete [] m_split_size;
  delete [] m_split_axis;
  delete [] m_split_position;
}

void Device::SplitData::resize(unsigned int size)
{

}
}

void SplitData::clear()
{
  m_split_first_indices.clear();
  m_split_size.clear();
  m_split_axis.clear();
  m_split_position.clear();
}

void SplitData::resize(unsigned int size)
{
  m_split_first_indices.resize(size);
  m_split_size.resize(size);
  m_split_axis.resize(size);
  m_split_position.resize(size);
}

__global__
void cu_split_small_nodes(Device::ChunkList chunklist, Device::SplitCandidates splitcan)
{
  __shared__ int num_prim;
  __shared__ int prim_starting_index;
  __shared__ int split_starting_index;
  __shared__ int node_index;

  int thread = threadIdx.x;
  if (thread == 0) // Load the shared variables
  {
    node_index = blockIdx.x;
    num_prim = chunklist.m_nodelist.m_node_prim_num[node_index];
    prim_starting_index = chunklist.m_nodelist.m_prim_starting_indices[node_index];
    split_starting_index = splitcan.m_split_data.m_split_offset[node_index];
  }
  __syncthreads();

  int current_split_axis;
  int current_split_index;
  int current_prim_index;
  float current_split_pos;
  long long left_prim_mask, right_prim_mask;
  __shared__ float3 prim_aabb_min_vert[64];
  __shared__ float3 prim_aabb_max_vert[64];

  // Evaluate whether a given primitive is on
  current_prim_index = prim_starting_index + thread;
  current_split_index = split_starting_index + thread;

  if (thread < num_prim)
  {
    prim_aabb_min_vert[thread] = chunklist.m_prim_aabbs.m_min_vert[current_prim_index];
    prim_aabb_max_vert[thread] = chunklist.m_prim_aabbs.m_max_vert[current_prim_index];
    left_prim_mask = right_prim_mask = 0;
  }
  __syncthreads();

  // TODO: Complete this kernel
}

void SplitCandidates::
split(ChunkList& list)
{
  // resize?

}
}
