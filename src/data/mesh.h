#pragma once

#include <map>
#include <thrust/device_vector.h>

#include "src/render/glincludes.h"
#include "src/data/aabb.h"
#include "ext/obj.h"

namespace CuKee
{

namespace Device {
struct Mesh
{
  float3* m_vbo;
  float3* m_nbo;
  int3* m_ibo;  // Do the paddings ourselves??
  Device::ArrAABB m_aabbs;
  int m_length;
};
}

/*
 * Use more buffer object to reduce copying.
 * Make this a light weight and renderable version of objloader's implementation
 */
class Mesh
{
public:
    Mesh(obj* mesh);
    Mesh() = delete;
    Mesh(const Mesh& rhs) = delete;

    ~Mesh();

    unsigned int size() const noexcept;
    void clear();
    void resize(unsigned int size);
    void compute_aabbs();

    // TODO: Evaluate whether these are neccessearyq...
    unsigned int get_vbo_size() const { return m_vbo_size; }
    unsigned int get_nbo_size() const { return m_nbo_size; }
    unsigned int get_cbo_size() const { return m_cbo_size; }
    unsigned int get_ibo_size() const { return m_ibo_size; }
    unsigned int get_tbo_size() const { return m_tbo_size; }

    const thrust::device_vector<glm::vec4>& get_vbo_readonly() const { return m_vbo; }
    const thrust::device_vector<glm::vec4>& get_nbo_readonly() const { return m_nbo; }
    const thrust::device_vector<glm::vec4>& get_cbo_readonly() const { return m_cbo; }
    const thrust::device_vector<glm::ivec4>& get_ibo_readonly() const { return m_ibo; }

    Device::Mesh to_device();

    AABB m_bounding_box;
private:
    unsigned int m_vbo_size; // Vertex buffer
    unsigned int m_nbo_size; // Normal buffer
    unsigned int m_cbo_size; // Color buffer
    unsigned int m_ibo_size; // Index buffer
    unsigned int m_tbo_size; // Texture buffer

    thrust::device_vector<glm::vec4> m_vbo;
    thrust::device_vector<glm::vec4> m_nbo;
    thrust::device_vector<glm::vec4> m_cbo;
    thrust::device_vector<glm::ivec4> m_ibo;
    thrust::device_vector<glm::vec2> m_tbo;
};
}

