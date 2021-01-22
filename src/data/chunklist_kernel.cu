#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "src/data/chunklist.h"

namespace CuKee
{
/*
 * Store the newly genereated vertices after clipping in a VertexBuffer
 */
struct Polygon
{
  glm::vec4 m_vertices[12];
  int m_size = 0;
};

// Create chunk from
  __global__
  void create_chunk(Device::ChunkList& chunklist)
  {
    int node_index = blockDim.x * blockIdx.x + threadIdx.x;
    int prim_starting_index = chunklist.m_nodelist.m_prim_starting_indices[node_index];
    int prim_num = chunklist.m_nodelist.m_node_prim_num[prim_num];

    // int num_chunk =
  }

  __global__
  void clip_prim(Device::Mesh& mesh, Device::SplitData& split_data)
  {
    // each thread is responsible for one primitive
    int prim_index = blockDim.x * blockIdx.x + threadIdx.x;

  }

  __host__  __device__
  void clip_line(glm::vec4& line_p1, glm::vec4& line_p2,
                 int& split_axis, float& split_pos, bool buffer_for_left, Polygon& vbuffer)
  {
    bool is_different_side = line_p1[split_axis] > split_pos ^ line_p2[split_axis] > split_pos;
    if (!is_different_side)
    {
      vbuffer.m_vertices[vbuffer.m_size] = line_p2;
      vbuffer.m_size += 1;
    }
    else
    {
      float split_ratio = (line_p1[split_axis] - split_pos) / (split_pos - line_p2[split_axis]);
      vbuffer.m_vertices[vbuffer.m_size] = line_p1 + (line_p2 - line_p1) * split_ratio;
      vbuffer.m_size += 1;
    }

    // What does buffer_for_left do?
  }

  __host__ __device__
  void clip_polygon(int& split_axis, float& split_pos, Polygon& polygon)
  {
    for(int i = 0; i < polygon.m_size; ++i)
    {

    }
  }

  __global__
  void sort_prim(Device::Mesh& mesh, Device::ChunkList& chunklist)
  {

  }

  // __device__

  __global__
  void count_prim_in_node(Device::ChunkList& chunklist)
  {
    int chunk_index = blockDim.x * blockIdx.x + threadIdx.x;

  }
}
