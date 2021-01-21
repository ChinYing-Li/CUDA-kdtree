#pragma once

#include <glm/glm.hpp>

#include <thrust/device_vector.h>


namespace CuKee
{

namespace Device
{
/****************************************************************************************
 * Struct of Array - AABB (GPU-based)
 */
struct ArrAABB
{
  float3* m_min_vert;
  float3* m_max_vert;
  int m_length;
};
}

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
  void set(unsigned int index, const float3  min_vert, const float3  max_vert);
  void set(unsigned int index, const float3* min_vert, const float3* max_vert);

  Device::ArrAABB to_device();

  AABB2iter begin();
  AABB2iter end();

  thrust::device_vector<float3> m_min_vert;
  thrust::device_vector<float3> m_max_vert;
};

/*
 * Reduce two AABB into one
 */
struct Reduce
{
  __device__
  AABBinstance operator()(const AABBinstance& lhs, const AABBinstance& rhs)
  {
    float3 lhs_min_vert = thrust::get<0>(lhs);
    float3 lhs_max_vert = thrust::get<1>(lhs);
    float3 rhs_min_vert = thrust::get<0>(rhs);
    float3 rhs_max_vert = thrust::get<1>(rhs);

    float3 max_vert;
    float3 min_vert;
    min_vert.x = thrust::min(lhs_min_vert.x, rhs_min_vert.x);
    min_vert.y = thrust::min(lhs_min_vert.y, rhs_min_vert.y);
    min_vert.z = thrust::min(lhs_min_vert.z, rhs_min_vert.z);
    max_vert.x = thrust::max(lhs_max_vert.x, rhs_max_vert.x);
    max_vert.y = thrust::max(lhs_max_vert.y, rhs_max_vert.y);
    max_vert.z = thrust::max(lhs_max_vert.z, rhs_max_vert.z);
    return thrust::make_tuple(min_vert, max_vert);
  }
};


}
