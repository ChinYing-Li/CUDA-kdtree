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
  m_bounding_box.m_min_vert = glm::vec4(bbox_vert[0], bbox_vert[1], bbox_vert[18], 0.0);
  m_bounding_box.m_max_vert = glm::vec4(bbox_vert[8], bbox_vert[5], bbox_vert[2], 0.0);
}

Mesh::~Mesh()
{}

unsigned int Mesh::size() const noexcept
{
  return m_vbo.size();
}

// TODO: implement rest of the methods for Mesh

DeviceMesh Mesh::to_device()
{
  DeviceMesh dmesh;
  dmesh.m_vbo = thrust::raw_pointer_cast(&m_vbo[0]);
}
} // namespace CuKee
