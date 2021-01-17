#pragma once

#include <cuda_runtime.h>

#include "src/data/aabb.h"
#include "src/data/mesh.h"
#include "src/data/nodelist.h"

#define MAX_PRIMITIVE_PER_CHUNK 256

namespace CuKee
{
struct DeviceChunkList
{
  DeviceArrAABB m_prim_aabbs;
  DeviceArrAABB m_chunk_aabbs;
  DeviceArrAABB m_node_aabbs;
  DeviceNodeList m_nodelist;
};

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
  void clip_prim(DeviceMesh& mesh, SplitList& splitlist);
  void count_prim_in_child();

  DeviceChunkList to_device();

private:
  unsigned int m_num_chunks;

  ArrAABB m_prim_aabbs;
  ArrAABB m_chunk_aabbs;
  ArrAABB m_node_aabbs;

  thrust::device_vector<int> m_chunk_size; // compute the start index of each chunk on the fly


};


}
