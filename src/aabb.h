#pragma once

#include <glm/glm.hpp>

#include <thrust/device_vector.h>

namespace CuKee
{
// Forward declaration
struct DeviceArrAABB;

/****************************************************************************************
 * Naive implementation of axis-aligned bounding box for CPU-based application
 */
class AABB
{
public:
  AABB();
  AABB(const glm::vec3 min_vert, const glm::vec3 max_vert);

  bool encloses(const AABB& box) const noexcept;
  unsigned int get_axis_with_max_distance() const noexcept;
  float get_surface_area() const noexcept;
  float max_vert_distance(const AABB& box) const noexcept;

  glm::vec3 m_min_vert;
  glm::vec3 m_max_vert;
};

/****************************************************************************************
 * Struct of Array - AABB (CPU-based)
 */

using AABBinstance = thrust::tuple<float3, float3>;
using f3iter = thrust::device_vector<float3>::iterator;
using AABBitertuple = thrust::tuple<f3iter, f3iter>;
using AABB2iter = thrust::zip_iterator<AABBitertuple>;

struct ArrAABB
{
  unsigned int size() const noexcept;
  void clear();
  void resize(unsigned int size);
  void copy(const ArrAABB& rhs);
  DeviceArrAABB to_device();

  AABB2iter begin();
  AABB2iter end();

  thrust::device_vector<float3> m_min_vert;
  thrust::device_vector<float3> m_max_vert;
};

/****************************************************************************************
 * Struct of Array - AABB (GPU-based)
 */
struct DeviceArrAABB
{
  float3* m_min_vert;
  float3* m_max_vert;
};
}
