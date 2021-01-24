#include <memory>

#include "mesh.h"
#include "objloader.h"

namespace CuKee {

struct glm_vec4_functor
{
  __host__ __device__
  glm::vec4 operator()(const glm::vec3& vec, const glm::vec3& empty_vec)
  {
    return glm::vec4(vec, 0.0);
  }
};

struct glm_ivec4_functor
{
  __host__ __device__
  glm::ivec4 operator()(const glm::ivec3& vec, const glm::ivec3& empty_vec)
  {
    return glm::ivec4(vec, 0.0);
  }
};

Mesh::
Mesh(obj* object):
  m_vbo_size(object->getVBOsize()),
  m_nbo_size(object->getNBOsize()),
  m_cbo_size(object->getCBOsize()),
  m_ibo_size(object->getIBOsize()),
  m_tbo_size(object->getTBOsize()),
  m_tbo(object->getTBO())
{
  /*
   * m_vbo(object->getVBO()),
    m_nbo(object->getNBO()),
    m_cbo(object->getCBO()),
    m_ibo(object->getIBO()),
    m_tbo(object->getTBO())
   */
  std::vector<float> bbox_vert = object->getBoundingBox();
  thrust::device_vector<glm::vec3> temp_vbo(object->getVBO());
  thrust::device_vector<glm::vec3> temp_nbo(object->getNBO());
  thrust::device_vector<glm::vec3> temp_cbo(object->getCBO());
  thrust::device_vector<glm::ivec3> temp_ibo(object->getIBO());

  m_vbo.reserve(m_vbo_size);
  m_nbo.reserve(m_nbo_size);
  m_cbo.reserve(m_cbo_size);
  m_ibo.reserve(m_ibo_size);

  thrust::fill(m_vbo.begin(), m_vbo.end(), glm::vec4(0.0));
  thrust::fill(m_nbo.begin(), m_nbo.end(), glm::vec4(0.0));
  thrust::fill(m_cbo.begin(), m_cbo.end(), glm::vec4(0.0));
  thrust::fill(m_ibo.begin(), m_ibo.end(), glm::ivec4(0));

  thrust::transform(temp_vbo.begin(), temp_vbo.end(), m_vbo.begin(), m_vbo.end(), glm_vec4_functor());
  thrust::transform(temp_nbo.begin(), temp_nbo.end(), m_nbo.begin(), m_nbo.end(), glm_vec4_functor());
  thrust::transform(temp_cbo.begin(), temp_cbo.end(), m_cbo.begin(), m_cbo.end(), glm_vec4_functor());
  thrust::transform(temp_ibo.begin(), temp_ibo.end(), m_ibo.begin(), m_ibo.end(), glm_ivec4_functor());

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

Device::Mesh Mesh::to_device()
{
  Device::Mesh dmesh;
  dmesh.m_length = m_vbo_size;
  dmesh.m_vbo
  dmesh.m_vbo = thrust::raw_pointer_cast(&m_vbo[0]);
  dmesh.m_ibo = thrust::raw_pointer_cast(&m_ibo[0]);
  return dmesh;
}
} // namespace CuKee
