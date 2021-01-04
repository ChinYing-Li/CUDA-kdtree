#pragma once

#include <glm/glm.hpp>

namespace CuKee
{
/*
 * Axis-aligned Bounding Box
 */
class AABB
{
public:
  AABB();
  AABB(const glm::vec3 min_vert, const glm::vec3 max_vert);
  bool encloses(const AABB& box);
  float max_vert_distance(const AABB& box);

  glm::vec3 m_min_vert;
  glm::vec3 m_max_vert;
};
}
