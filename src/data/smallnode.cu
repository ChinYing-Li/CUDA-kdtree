// CUDA Dependencies
#include <cuda.h>

// GL dependencies
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "smallnode.h"

namespace CuKee
{
void SmallNodeList::
resize(unsigned int node_size)
{
  m_prim_bits.resize(node_size);
  m_small_root.resize(node_size);
  m_root_index.resize(node_size);
}

Device::SmallNodeList SmallNodeList::
to_device()
{
  Device::SmallNodeList dev_list;
  dev_list.m_prim_bits = thrust::raw_pointer_cast(&m_prim_bits[0]);
  dev_list.m_root_index = thrust::raw_pointer_cast(&m_root_index[0]);
  dev_list.m_small_root = thrust::raw_pointer_cast(&m_small_root[0]);
  dev_list.m_node_list = NodeList::to_device();
  return dev_list;
}

void SmallNodeList::
split()
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
  __shared__ glm::vec3 prim_min_vert[64];
  __shared__ glm::vec3 prim_max_vert[64];

  int split_offset = threadIdx.x;
  if (split_offset == 0)
  {
    node_index = blockIdx.x;
    node_split_starting_index = splitcan.m_split_data.m_split_offset[node_index];
    num_prim = activelist.m_node_list.m_node_prim_num[node_index];
    prim_starting_index = activelist.m_node_list.m_prim_starting_indices[node_index];
    cudaMemcpy(prim_min_vert, &activelist.m_node_list.m_node_aabb.m_min_vert[prim_starting_index],
               num_prim * sizeof (glm::vec4), cudaMemcpyDeviceToDevice);
    cudaMemcpy(prim_max_vert, &activelist.m_node_list.m_node_aabb.m_max_vert[prim_starting_index],
               num_prim * sizeof (glm::vec4), cudaMemcpyDeviceToDevice);

  }
 __syncthreads();

 float is_valid_offset = (split_offset >= splitcan.m_split_data.m_split_size[node_index]);
 int split_axis = splitcan.m_split_data.m_split_axis[node_split_starting_index + split_offset] * is_valid_offset;
 int split_pos  = splitcan.m_split_data.m_split_position[node_split_starting_index + split_offset] * is_valid_offset;

 // Iterate through all the primitives (triangles)
 for (int i = 0; i < num_prim; ++i)
 {
   bool is_on_left = prim_max_vert[i][split_axis] < split_pos;
   bool is_on_right = prim_min_vert[i][split_axis] > split_pos;
   splitcan.m_left_prim_bitmask[node_split_starting_index + split_offset] |= is_on_left << i;
   splitcan.m_right_prim_bitmask[node_split_starting_index + split_offset] |= is_on_right << i;
 }
  __syncthreads();
}

__global__
void process(Device::SmallNodeList& activelist,
             Device::SmallNodeList& nextlist)
{

}

__device__
float compute_node_area(const glm::vec4& min_vert,
                        const glm::vec4& max_vert)
{
  glm::vec4 diff = max_vert - min_vert;
  return 2.0 * (diff[0] * diff[1] + diff[1] * diff[2] + diff[2] * diff[0]);
}

/*
 * Each block is responsible for one node, each thread process one splitplane
 */
__device__
void compute_SAH(Device::SmallNodeList activelist,
                 Device::SplitCandidates splitcan)
{
  int thread = threadIdx.x;
  __shared__ int node_index;
  __shared__ int root_index;
  __shared__ long long node_prim_bitmask;
  __shared__ long long root_prim_bitmask;
  __shared__ int node_split_starting_index;
  __shared__ int num_split;
  __shared__ float node_area;
  __shared__ float root_SAH;
  __shared__ float root_SAH_INV;
  extern __shared__ float SAH_arr[64];

  if (thread == 0)
  {
    node_index = blockIdx.x;
    root_index = activelist.m_root_index[node_index];
    node_prim_bitmask = activelist.m_prim_bits[node_index];
    root_prim_bitmask = activelist.m_prim_bits[root_index];
    node_split_starting_index = activelist.m_node_list.m_split_data.m_split_offset[node_index];
    num_split = activelist.m_node_list.m_split_data.m_split_size[node_index];
    glm::vec4 diff = activelist.m_node_list.m_node_aabb.m_max_vert[node_index]
        - activelist.m_node_list.m_node_aabb.m_min_vert[node_index];
    node_area = 2.0 * (diff[0] * diff[1] + diff[1] * diff[2] + diff[2] * diff[0]);
    root_SAH = __popcll(root_prim_bitmask);
    root_SAH_INV = 1.0 / root_SAH;
  }
  __syncthreads();


  if (thread < num_split)
  {
    int split_index = node_split_starting_index + thread;
    float cost_left = __popcll(root_prim_bitmask & splitcan.m_left_prim_bitmask[split_index]);
    float cost_right = __popcll(root_prim_bitmask & splitcan.m_right_prim_bitmask[split_index]);
    // TODO: calculate the area
    float area_left = 0.0;
    float area_right = 0.0;
    SAH_arr[thread] = (cost_left * area_left + cost_right * area_right) * root_SAH_INV + 0.0;
  }
}
}
