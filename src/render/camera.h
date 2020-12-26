#pragma once

#include <math.h>
#include <iostream>
#include "glincludes.h"

namespace Cluster {

enum CamMove
{
    CM_move_forward  = 3,
    CM_move_backward = 4,
    CM_turn_cw       = 5,
    CM_turn_ccw      = 6
};

class Camera
{
public:
    Camera();
    Camera(const float x, const float y, const float z);
    Camera(const glm::vec3 position);
    ~Camera() = default;

    void update_project_transform(glm::mat4& mat) const noexcept;
    void update_view_transform(glm::mat4& mat) const noexcept;

    void set_aspect_ratio(float aspect_ratio);
    float get_aspect_ratio() const noexcept;

    // In degrees
    void set_fovy(float fovy);
    float get_fovy() const noexcept;

    float get_near_plane() const noexcept;
    float get_far_plane() const noexcept;
    void set_near_plane(float near);
    void set_far_plane(float far);

    void set_eye(glm::vec3 eye);
    void set_direction(glm::vec3 direction);
    void set_up(glm::vec3 up);

private:
    float m_aspect_ratio = 1.0f;
    float m_fovy;
    float m_near_plane = 0.1f;
    float m_far_plane = 100.0f;
    glm::vec2 m_screenspace_bottomleft_coordinate = glm::vec2(-1);
    glm::vec2 m_screenspace_top_coordinate = glm::vec2(1);

    glm::vec3 m_eye;
    glm::vec3 m_direction;
    glm::vec3 m_up;
};
}
