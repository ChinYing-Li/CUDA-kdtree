#include <device_launch_parameters.h>

#include "nodelist.h"

namespace CuKee
{
NodeList::NodeList()
{

}

inline void NodeList::
clear()
{
  m_num_node = m_num_prim = 0;
  m_left_child_indices.clear();
  m_right_child_indices.clear();
  m_starting_indices_in_prim.clear();
  m_node_prim_num.clear();
  m_prim_indices.clear();
  m_splitdata.clear();
}

inline void NodeList::
resize_node(unsigned int size)
{
  m_num_node = size;
  m_left_child_indices.resize(size);
  m_right_child_indices.resize(size);
  m_starting_indices_in_prim.resize(size);
  m_node_prim_num.resize(size);
  m_node_aabbs.resize(size);
  m_splitdata.resize(size);
}

inline void NodeList::
resize_prim(unsigned int size)
{
  m_num_prim = size;
  m_prim_indices.resize(size);
  m_prim_aabbs.resize(size);
}

/****************************************************************************************
 * Remove the empty space in large nodes
 */

__global__
void cut_empty_space()
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  // update the aabb of the node
}

void NodeList::
cut_empty_space()
{

}

/*
 * Split large nodes into child nodes
 */
void NodeList::
split_node()
{

}

inline Device::NodeList NodeList::
to_device()
{
  Device::NodeList dev_list;
  dev_list.m_left_child_indices = thrust::raw_pointer_cast(&m_left_child_indices[0]);
  dev_list.m_right_child_indices = thrust::raw_pointer_cast(&m_right_child_indices[0]);
  dev_list.m_prim_starting_indices = thrust::raw_pointer_cast(&m_starting_indices_in_prim[0]);
  dev_list.m_node_prim_num = thrust::raw_pointer_cast(&m_node_prim_num[0]);
  dev_list.m_prim_indices = thrust::raw_pointer_cast(&m_prim_indices[0]);
  dev_list.m_prim_aabb = m_prim_aabbs.to_device();
  dev_list.m_node_aabb = m_node_aabbs.to_device();
  // TODO: m_num_node & m_num_prim
}


}
