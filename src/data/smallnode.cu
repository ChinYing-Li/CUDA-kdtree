#include <device_launch_parameters.h>

#include "smallnode.h"

namespace CuKee
{
void SmallNodeList::resize(unsigned int node_size)
{
  m_prim_bits.resize(node_size);
  m_small_root.resize(node_size);
  m_root_index.resize(node_size);
}



Device::SmallNodeList SmallNodeList::to_device()
{
  Device::SmallNodeList dev_list;
  dev_list.m_prim_bits = thrust::raw_pointer_cast(&m_prim_bits[0]);
  dev_list.m_root_index = thrust::raw_pointer_cast(&m_root_index[0]);
  dev_list.m_small_root = thrust::raw_pointer_cast(&m_small_root[0]);
  dev_list.m_node_list = NodeList::to_device();
  return dev_list;
}

void SmallNodeList::split()
{
  dim3 grid_dim(m_num_node, 1, 1);
  dim3 block_dim(6 * MAX_PRIM_PER_SMALL_NODE, 1, 1);
}

/*
 * The kernels used in Algorithm 4 "Small Node Stage"
 */
__global__
void preprocess(Device::SmallNodeList& activelist, Device::SplitCandidates& splitcan)
{
  // each thread processes one potential split in the node
  // each block processes one small node's split candidates
  __shared__ int node_index;
  __shared__ int node_split_starting_index;
  __shared__ int num_prim;
  __shared__ int prim_starting_index;
  __shared__ float3 prim_min_vert[64];
  __shared__ float3 prim_max_vert[64];

  int split_offset = threadIdx.x;
  if (split_offset == 0)
  {
    node_index = blockIdx.x;
    node_split_starting_index = splitcan.m_split_data.m_split_offset[node_index];
    num_prim = activelist.m_node_list.m_node_prim_num[node_index];
    prim_starting_index = activelist.m_node_list.m_prim_starting_indices[node_index];
    cudaMemcpy(prim_min_vert, &activelist.m_node_list.m_node_aabb.m_min_vert[prim_starting_index],
               num_prim * sizeof (float3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(prim_max_vert, &activelist.m_node_list.m_node_aabb.m_max_vert[prim_starting_index],
               num_prim * sizeof (float3), cudaMemcpyDeviceToDevice);

  }
 __syncthreads();

 if (split_offset >= splitcan.m_split_data.m_split_size[node_index]) return;
 int split_axis = splitcan.m_split_data.m_split_axis[node_split_starting_index + split_offset];
 int split_pos  = splitcan.m_split_data.m_split_position[node_split_starting_index + split_offset];

 // Iterate through all the primitives (triangles)
 for (int i = 0; i < num_prim; ++i)
 {
   bool is_on_left = prim_max_vert[i][split_axis] ;
 }

 splitcan.m_left_prim_bitmask[node_split_starting_index + split_offset] = 0;
 splitcan.m_right_prim_bitmask[node_split_starting_index + split_offset] = 0;

}

__global__
void compute_SAH(Device::SmallNodeList activelist, Device::SplitCandidates splitcan)
{

}
}
