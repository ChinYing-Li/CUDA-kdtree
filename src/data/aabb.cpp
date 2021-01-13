#include <thrust/copy.h>

#include "aabb.h"

namespace CuKee
{
/****************************************************************************************
 * AABB
 */
AABB::
AABB():
  m_min_vert(0.0f),
  m_max_vert(0.0f)
{}

AABB::
AABB(const glm::vec3 min_vert, const glm::vec3 max_vert):
  m_min_vert(min_vert),
  m_max_vert(max_vert)
{}

/*
 *
 */
bool AABB::
encloses(const AABB& box) const noexcept
{
  bool res = m_min_vert.x <= box.m_min_vert.x;
  res &= (m_min_vert.y <= box.m_min_vert.y);
  res &= (m_min_vert.z <= box.m_min_vert.z);
  res &= (m_max_vert.x <= box.m_max_vert.x);
  res &= (m_max_vert.y <= box.m_max_vert.y);
  res &= (m_max_vert.z <= box.m_max_vert.z);
  return res;
}

/*
 * Return the axis that spans the most.
 * x axis => 0
 * y axis => 1
 * z axis => 2
 */
unsigned int AABB::
get_axis_with_max_distance() const noexcept
{
  glm::vec3 diagonal = m_max_vert - m_min_vert;
  if (diagonal.x > diagonal.y)
  {
    if (diagonal.x > diagonal.z) return 0;
    return 2;
  }
  else
  {
    if (diagonal.y > diagonal.z) return 1;
    return 2;
  }
}

/*
 * Returns the surface area of the bounding box
 */
float AABB::
get_surface_area() const noexcept
{
  glm::vec3 diagonal = m_max_vert - m_min_vert;
  return 2.0 * (diagonal.x * diagonal.y + diagonal.y * diagonal.z + diagonal.z * diagonal.x);
}

/* Get the maximum difference between this object and box, along a
 * specified axis. set axis to some negative value to evaluate overall
 * all three axis.
 */
float AABB::
max_vert_distance(const AABB &box) const noexcept
{
  float res = std::max(std::abs(m_min_vert.x - box.m_min_vert.x),
                       std::abs(m_min_vert.y - box.m_min_vert.y));
  res = std::max(res, std::abs(m_min_vert.z - box.m_min_vert.z));
  res = std::max(res, std::abs(m_max_vert.x - box.m_max_vert.x));
  res = std::max(res, std::abs(m_max_vert.y - box.m_max_vert.y));
  res = std::max(res, std::abs(m_max_vert.z - box.m_max_vert.z));
  return res;
}

/****************************************************************************************
 * ArrAABB
 */

inline unsigned int ArrAABB::
size() const noexcept
{
  return m_max_vert.size();
}

inline void ArrAABB::
clear()
{
  m_min_vert.clear();
  m_max_vert.clear();
}

inline void ArrAABB::
resize(unsigned int size)
{
  m_min_vert.resize(size);
  m_max_vert.resize(size);
}

inline void ArrAABB::
copy(const ArrAABB &rhs)
{
  this->resize(rhs.size());
  thrust::copy(m_min_vert.begin(), m_min_vert.end(), rhs.m_min_vert.begin());
  thrust::copy(m_max_vert.begin(), m_max_vert.end(), rhs.m_max_vert.begin());
}


}
