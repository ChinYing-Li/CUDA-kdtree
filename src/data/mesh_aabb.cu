#include <cuda_runtime.h>

#include "math/vector_type.h"
#include "geometry/mesh_aabb.h"

namespace CuKee
{
  __global__
  void krnl_triangle_aabb(DeviceMesh mesh)
  {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= mesh.m_length)
    {
      return;
    }

    float3 min_vert, max_vert;
    float3_minimum_op min_op;
    float3_maximum_op max_op;
    // int mesh_index = 3 * index;
    min_vert = min_op(mesh.m_vbo[mesh.m_ibo[index].x], mesh.m_vbo[mesh.m_ibo[index].y]);
    min_vert = min_op(min_vert, mesh.m_vbo[mesh.m_ibo[index].z]);
    max_vert = max_op(mesh.m_vbo[mesh.m_ibo[index].x], mesh.m_vbo[mesh.m_ibo[index].y]);
    max_vert = max_op(max_vert, mesh.m_vbo[mesh.m_ibo[index].z]);
    mesh.m_aabbs.m_min_vert[index] = min_vert;
    mesh.m_aabbs.m_max_vert[index] = max_vert;
  }
}
