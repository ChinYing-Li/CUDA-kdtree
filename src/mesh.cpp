#include <memory>

#include "mesh.h"
#include "objloader.h"

namespace CuKee {
Mesh::
Mesh(obj* object):
  m_vbo_size(object->getVBOsize()),
  m_nbo_size(object->getNBOsize()),
  m_cbo_size(object->getCBOsize()),
  m_ibo_size(object->getIBOsize()),
  m_tbo_size(object->getTBOsize()),
  m_vbo(object->getVBO()),
  m_nbo(object->getNBO()),
  m_cbo(object->getCBO()),
  m_ibo(object->getIBO()),
  m_tbo(object->getTBO())
{
  std::vector<float> bbox_vert = object->getBoundingBox();
  m_bounding_box.m_min_vert = glm::vec3(bbox_vert[0], bbox_vert[1], bbox_vert[18]);
  m_bounding_box.m_max_vert = glm::vec3(bbox_vert[8], bbox_vert[5], bbox_vert[2]);
}

Mesh::~Mesh()
{}

} // namespace Cluster
