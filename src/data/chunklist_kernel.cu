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

/*
 *
 */
  __global__
  void create_chunk(Device::ChunkList& chunklist,
                    int num_node)
  {
    int node_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (node_index >= num_node) return;

    int prim_starting_index = chunklist.m_nodelist.m_prim_starting_indices[node_index];
    int prim_num = chunklist.m_nodelist.m_node_prim_num[node_index];

    // int num_chunk =
  }

  /*
   * Then we use scan to get chunk_offset from chunk_per_node
   */
  __host__ __device__
  void get_chunk_per_node(int* prim_num_per_node,
                          int num_node,
                          int* chunk_per_node)
  {
    int node_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (node_index >= num_node) return;
    int prim_num = prim_num_per_node[node_index];
    chunk_per_node[node_index] = (prim_num >> MAX_PRIM_PER_CHUNK_LOG2) + bool(prim_num & ((1 << 8) - 1));
  }

  __global__
  void clip_prim(Device::Mesh& mesh, Device::SplitData& split_data)
  {
    // each thread is responsible for one primitive
    int prim_index = blockDim.x * blockIdx.x + threadIdx.x;

  }

  /*
   * int
   */
  __host__  __device__
  void clip_line(glm::vec4& line_p1,
                 glm::vec4& line_p2,
                 int& split_axis,
                 float& split_pos,
                 SplitSide side,
                 Polygon& vbuffer)
  {
    bool p1_at_side = float(side) * (line_p1[split_axis] - split_pos) >= 0.0;
    bool p2_at_side = float(side) * (line_p2[split_axis] - split_pos) >= 0.0;
    if (p1_at_side && p2_at_side)
    {
      vbuffer.m_vertices[vbuffer.m_size] = line_p2;
      vbuffer.m_size += 1;
    }

    if (p1_at_side != p2_at_side)
    {
      float split_ratio = (line_p1[split_axis] - split_pos) / (split_pos - line_p2[split_axis]);
      vbuffer.m_vertices[vbuffer.m_size] = line_p1 + (line_p2 - line_p1) * split_ratio;
      vbuffer.m_size += 1;
    }
  }

  __host__ __device__
  void clip_polygon(int& split_axis,
                    float& split_pos,
                    SplitSide side,
                    Polygon& old_polygon,
                    Polygon& new_polygon)
  {
    int i;
    for(i = 0; i < old_polygon.m_size - 1; ++i)
    {
      clip_line(old_polygon.m_vertices[i], old_polygon.m_vertices[i+1],
          split_axis, split_pos, side, new_polygon);
    }
    clip_line(old_polygon.m_vertices[old_polygon.m_size - 1], old_polygon.m_vertices[0],
        split_axis, split_pos, side, new_polygon);
  }

  __host__ __device__
  void clip_polygon_to_aabb()
  {

  }


  /*
   * sort_result is an array of size 2*num_prim, where num_prim is the number of primitives
   * in activelist's corresponding mesh.
   */
  __global__
  void sort_prim(Device::ChunkList& activelist,
                 int* sort_result)
  {
    /* Variables shared across the thread block */
    __shared__ int split_axis;
    __shared__ float split_pos;
    __shared__ unsigned int prim_starting_index;
    __shared__ unsigned int num_prim;
    __shared__ unsigned int node_index;
    __shared__ unsigned int chunk_index;

    unsigned int thread = threadIdx.x;
    unsigned int index = blockDim.x * blockIdx.x + thread;
    if (thread == 0)
    {
      chunk_index = blockIdx.x;
      node_index = activelist.m_node_indices[chunk_index];
      num_prim = activelist.m_num_prim_per_chunk[chunk_index];
      prim_starting_index = activelist.m_chunk_prim_starting_indices[chunk_index];
      split_axis = activelist.m_nodelist.m_split_data.m_split_axis[node_index];
      split_pos = activelist.m_nodelist.m_split_data.m_split_position[node_index];
    }
    __syncthreads();

    glm::vec4 prim_min_vert;
    glm::vec4 prim_max_vert;
    int prim_index;

    if (thread < num_prim)
    {
      prim_index = prim_starting_index + thread;
      prim_min_vert = activelist.m_nodelist.m_prim_aabb.m_min_vert[prim_index];
      prim_max_vert = activelist.m_nodelist.m_prim_aabb.m_max_vert[prim_index];
      sort_result[2 * prim_index    ] = prim_max_vert[split_axis] > split_pos || prim_min_vert[split_axis] >= split_pos;
      sort_result[2 * prim_index + 1] = prim_max_vert[split_axis] <= split_pos || prim_min_vert[split_axis] < split_pos;
    }
  }

  /*
   * tight_node_aabb is computed in the previous step
   */
  __global__
  void get_empty_space_removal(Device::ArrAABB& tight_node_aabb,
                               Device::ChunkList& activelist,
                               int num_node,
                               int cut_axis,
                               int* cut_bitmask)
  {
    int node_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (node_index >= num_node) return;

    float tight_aabb_min_pos = tight_node_aabb.m_min_vert[node_index][cut_axis];
    float tight_aabb_max_pos = tight_node_aabb.m_max_vert[node_index][cut_axis];
    float curr_aabb_min_pos = activelist.m_nodelist.m_node_aabb.m_min_vert[node_index][cut_axis];
    float curr_aabb_max_pos = activelist.m_nodelist.m_node_aabb.m_max_vert[node_index][cut_axis];

    float distance_thres = (curr_aabb_max_pos - curr_aabb_min_pos) * EMPTY_REMOVAL_THRES;
    bool is_above_thres = (curr_aabb_max_pos - tight_aabb_max_pos) > distance_thres;
    cut_bitmask[node_index] |= int(is_above_thres) << (2 * cut_axis);

    is_above_thres = (tight_aabb_min_pos - curr_aabb_min_pos) > distance_thres;
    cut_bitmask[node_index] |= int(is_above_thres) << (2 * cut_axis + 1);
  }

  /*
   * Each thread clips a primitive a.k.a. triangle
   * Each block corresponds to one chunk in activelist
   */
  __global__
  void clip_prims(Device::ChunkList& activelist,
                  int num_current_node,
                  int* current_split_axis,
                  int* current_split_pos,
                  Device::Mesh& mesh,
                  int n_left) // What's n_left
  {
    int thread = threadIdx.x;
    __shared__ int chunk_index;

    if (thread == 0)
    {
      chunk_index = blockIdx.x;
    }
    __syncthreads();

    __shared__ int split_axis;
    __shared__ float split_pos;
  }

  /*
   * After computing on activelist, put our result into nextlist
   */
  __global__
  void update_aabb(Device::ChunkList& activelist,
                   Device::ChunkList& nextlist,
                   int num_node)
  {
    int node_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (node_index >= num_node)
    {
      return;
    }

    glm::vec4 left_min_vert, left_max_vert, right_min_vert, right_max_vert;
    left_min_vert = right_min_vert =
        activelist.m_nodelist.m_node_aabb.m_min_vert[node_index];
    left_max_vert = right_max_vert =
        activelist.m_nodelist.m_node_aabb.m_max_vert[node_index];

    int split_axis = activelist.m_nodelist.m_split_data.m_split_axis[node_index];
    float split_pos = activelist.m_nodelist.m_split_data.m_split_position[node_index];
    left_max_vert[split_axis] = right_min_vert[split_axis] = split_pos;
    // Parent aabb ???
    nextlist.m_nodelist.m_parent_node_aabb.m_min_vert[node_index] = left_min_vert;
    nextlist.m_nodelist.m_parent_node_aabb.m_max_vert[node_index] = left_max_vert;
    nextlist.m_nodelist.m_parent_node_aabb.m_min_vert[node_index + num_node] = right_min_vert;
    nextlist.m_nodelist.m_parent_node_aabb.m_max_vert[node_index + num_node] = right_max_vert;

    int current_depth = activelist.m_nodelist.m_node_depth[node_index];

    // Bank conflict over here! How can we optimize this?
    nextlist.m_nodelist.m_node_depth[node_index] = nextlist.m_nodelist.m_node_depth[node_index + num_node]
        = current_depth + 1;
  }

  __global__
  void count_prim_in_node(Device::ChunkList& chunklist)
  {
    int chunk_index = blockDim.x * blockIdx.x + threadIdx.x;

  }
}
