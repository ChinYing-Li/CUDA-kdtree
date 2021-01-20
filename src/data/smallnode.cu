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



DeviceSmallNodeList SmallNodeList::to_device()
{
  DeviceSmallNodeList dev_list;
  dev_list.m_prim_bits = thrust::raw_pointer_cast(&m_prim_bits[0]);
  dev_list.m_root_index = thrust::raw_pointer_cast(&m_root_index[0]);
  dev_list.m_small_root = thrust::raw_pointer_cast(&m_small_root[0]);
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
void preprocess(Device::SmallNodeList& activelist, DeviceSplitCandidates& splitcan)
{
  __shared__ int node_index;
  __shared__ int node_split_starting_index;

  int split_offset = threadIdx.x;
  if (split_offset == 0)
  {
    node_index = blockIdx.x;
    node_split_starting_index = splitcan.m_split_data.m_split_offset[node_index];
  }
 __syncthreads();


 splitcan.m_left_prim_bitmask[node_split_starting_index + split_offset] = 0;
}

__global__
void compute_SAH(Device::SmallNodeList activelist, DeviceSplitCandidates splitcan)
{

}
}
