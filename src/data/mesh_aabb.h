#pragma once

#include "mesh.h"

namespace CuKee
{
  void get_triangle_aabb(DeviceMesh& mesh, unsigned int length);

  __global__
  void krnl_triangle_aabb(DeviceMesh mesh);
}
