#include "chunklist.h"

namespace CuKee
{
ChunkList::
ChunkList()
{

}

void ChunkList::
create_chunks_from_nodes()
{

}

/*
 * Reduce node aabbs in each chunk to get per-chunk aabb
 */
void ChunkList::
chunk_reduce_aabb()
{

}

/*
 * Determine which child node the primitive belongs to.
 */
void ChunkList::
sort_prim(thrust::device_vector<bool>& res)
{

}

/*
 *
 */
void ChunkList::
clip_prim(Device::Mesh& mesh, Device::SplitData& split_data)
{

}

/*
 * Count primitives in each child node.
 */
void ChunkList::
count_prim_in_child()
{

}

Device::ChunkList ChunkList::to_device()
{
  Device::ChunkList dev_list;
  dev_list.m_node_chunk_starting_indices = thrust::raw_pointer_cast(&m_node_chunk_starting_indices[0]);
  dev_list.m_chunk_aabbs = m_chunk_aabbs.to_device();
  dev_list.m_num_prim_per_chunk = thrust::raw_pointer_cast(&m_num_prim_per_chunk[0]);
  dev_list.m_chunk_prim_starting_indices = thrust::raw_pointer_cast(&m_chunk_prim_starting_indices[0]);
  dev_list.m_nodelist = NodeList::to_device();
  return dev_list;
}
}

