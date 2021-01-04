#include "aabb.h"

namespace CuKee
{
AABB::AABB():
  m_min_vert(0.0f),
  m_max_vert(0.0f)
{}

AABB::AABB(const glm::vec3 min_vert, const glm::vec3 max_vert):
  m_min_vert(min_vert),
  m_max_vert(max_vert)
{}

/*
 *
 */
bool AABB::
encloses(const AABB& box)
{
  bool res = m_min_vert.x <= box.m_min_vert.x;
  res &= (m_min_vert.y <= box.m_min_vert.y);
  res &= (m_min_vert.z <= box.m_min_vert.z);
  res &= (m_max_vert.x <= box.m_max_vert.x);
  res &= (m_max_vert.y <= box.m_max_vert.y);
  res &= (m_max_vert.z <= box.m_max_vert.z);
  return res;
}

/* Get the maximum difference between this object and box, along a
 * specified axis. set axis to some negative value to evaluate overall
 * all three axis.
 */
float AABB::
max_vert_distance(const AABB &box)
{
  float res = std::max(std::abs(m_min_vert.x - box.m_min_vert.x),
                       std::abs(m_min_vert.y - box.m_min_vert.y));
  res = std::max(res, std::abs(m_min_vert.z - box.m_min_vert.z));
  res = std::max(res, std::abs(m_max_vert.x - box.m_max_vert.x));
  res = std::max(res, std::abs(m_max_vert.y - box.m_max_vert.y));
  res = std::max(res, std::abs(m_max_vert.z - box.m_max_vert.z));
  return res;
}
}
