#pragma once

#include <cuda_runtime.h>

#include "src/data/aabb.h"
#include "src/data/mesh.h"
#include "src/data/nodelist.h"

#define MAX_PRIMITIVE_PER_CHUNK 256

namespace CuKee
{
namespace Device
{
struct ChunkList
{
  int* m_node_chunk_starting_indices;
  int* m_num_chunk_per_node;
  int* m_chunk_prim_starting_indices;
  int* m_num_prim_per_chunk;
  int* m_result_keys;
  Device::ArrAABB m_chunk_aabbs;
  Device::NodeList m_nodelist;
};
}


class ChunkList final : public NodeList
{
public:
  ChunkList();
  ~ChunkList();
  void clear() override;
  void resize_node(unsigned int size) override;
  void resize_prim(unsigned int size) override;
  void resize_chunk(unsigned int size);

  void create_chunks_from_nodes();
  void chunk_reduce_aabb();
  void sort_prim(thrust::device_vector<bool>& res);
  void clip_prim(Device::Mesh& mesh, Device::SplitData& split_data);
  void count_prim_in_child();

  Device::ChunkList to_device();

private:
  unsigned int m_num_chunks;
  ArrAABB m_chunk_aabbs;

  // per node members
  thrust::device_vector<int> m_node_chunk_starting_indices; // The index of the corresponding
  thrust::device_vector<int> m_num_chunk_per_node; // compute the start index of each chunk on the fly
  // per chunk members
  thrust::device_vector<int> m_num_prim_per_chunk;
  thrust::device_vector<int> m_chunk_prim_starting_indices;
  thrust::device_vector<int> m_result_keys;
};


}
