#pragma once

#include <cuda_runtime.h>

#include "aabb.h"
#include "mesh.h"

#define MAX_TRIANGLE_PER_CHUNK 256

namespace CuKee
{

class ChunkList
{
public:
  ChunkList();
  ~ChunkList();

  

private:
  ArrAABB m_triangle_aabbs;
  ArrAABB m_chunk_aabbs;
  ArrAABB m_node_aabbs;

  thrust::device_vector<int> m_chunk_size; // compute the start index of each chunk on the fly
  //
  // thrust::device_vector<int>
};

struct DeviceChunkList
{

};
}
