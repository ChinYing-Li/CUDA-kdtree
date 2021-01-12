#pragma once

#include <map>
#include <thrust/device_vector.h>

#include "glincludes.h"
#include "aabb.h"
#include "obj.h"

namespace CuKee
{

struct DeviceMesh;

/*
 * Use more buffer object to reduce copying.
 * Make this a light weight and renderable version of objloader's implementation
 */
class Mesh
{
public:
    Mesh(obj* mesh);
    Mesh() = delete;
    ~Mesh();

    unsigned int get_vbo_size() const { return m_vbo_size; }
    unsigned int get_nbo_size() const { return m_nbo_size; }
    unsigned int get_cbo_size() const { return m_cbo_size; }
    unsigned int get_ibo_size() const { return m_ibo_size; }
    unsigned int get_tbo_size() const { return m_tbo_size; }

    const thrust::device_vector<float>& get_vbo_readonly() const { return m_vbo; }
    const thrust::device_vector<float>& get_nbo_readonly() const { return m_nbo; }
    const thrust::device_vector<float>& get_cbo_readonly() const { return m_cbo; }
    const thrust::device_vector<int>& get_ibo_readonly() const { return m_ibo; }

    DeviceMesh to_device();

    AABB m_bounding_box;
private:
    unsigned int m_vbo_size; // Vertex buffer
    unsigned int m_nbo_size; // Normal buffer
    unsigned int m_cbo_size; // Color buffer
    unsigned int m_ibo_size; // Index buffer
    unsigned int m_tbo_size; // Texture buffer

    thrust::device_vector<float> m_vbo;
    thrust::device_vector<float> m_nbo;
    thrust::device_vector<float> m_cbo;
    thrust::device_vector<int> m_ibo;
    thrust::device_vector<float> m_tbo;
};

struct DeviceMesh
{
  float* m_vbo;
  float3* m_nbo;
  int3* m_ibo;
  int m_length;
};
}

