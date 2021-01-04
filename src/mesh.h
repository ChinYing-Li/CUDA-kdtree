#pragma once

#include <map>
#include "glincludes.h"
#include "aabb.h"
#include "obj.h"

namespace CuKee
{

/*
 * TODO: refactor Mesh; do not inherit from GLObject.
 * Use more buffer object to reduce copying.
 * Make this a light weight and renderable version of objloader's implementation?
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

    const std::vector<float>& get_vbo_readonly() const { return m_vbo; }
    const std::vector<float>& get_nbo_readonly() const { return m_nbo; }
    const std::vector<float>& get_cbo_readonly() const { return m_cbo; }
    const std::vector<int>& get_ibo_readonly() const { return m_ibo; }

    AABB m_bounding_box; // Perhaps make this a unique_ptr as well...but a lot of overhead
private:
    unsigned int m_vbo_size; // Vertex buffer
    unsigned int m_nbo_size; // Normal buffer
    unsigned int m_cbo_size; // Color buffer
    unsigned int m_ibo_size; // Index buffer
    unsigned int m_tbo_size; // Texture buffer

    std::vector<float> m_vbo;
    std::vector<float> m_nbo;
    std::vector<float> m_cbo;
    std::vector<int> m_ibo;
    std::vector<float> m_tbo;
};
}

