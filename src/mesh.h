#pragma once

#include "globject.h"
// Forward declarations
namespace objl
{
class Loader; class Material; class Mesh;
}

namespace Cluster{
class GameData;

/*
 *
 */
class Mesh final : public GLObejct
{
public:
    Mesh(objl::Mesh& mesh, std::shared_ptr<GameData> data_ptr, unsigned int num_instance);
    ~Mesh() = default;

    void draw(GLuint& shaderID) override;

    std::string name;
    void send_instance_matrices(std::vector<glm::mat4>& instance_models) override;

private:
    unsigned int m_numinstance;
    void init(objl::Mesh& mesh);
    void draw_textures(GLuint shaderID);
    void set_material_uniform(GLuint& shaderID);

    objl::Material* m_material_ptr = nullptr;
    std::map<std::string, std::shared_ptr<Texture>> map_ptrs;
    std::vector<bool> use_maps;

/*map_Ka_ptr -->GL_TEXTURE0
map_Kd_ptr = nullptr; GL_TEXTURE1
map_Ks_ptr = nullptr; GL_TEXTURE2
map_Ns_ptr = nullptr;GL_TEXTURE3
map_d_ptr = nullptr; GL_TEXTURE4
map_bump_ptr = nullptr; GL_TEXTURE5*/
};

} // namespace Cluster

