
#include "camera.h"

namespace Cluster
{
Camera::Camera():
m_eye(0.0f, 0.5f, 0.0f),
m_direction(1.0f, 0.0f, -0.1f),
m_up(0, 1, 0)
{}

Camera::Camera(const float x, const float y, const float z):
m_eye(x, y, z),
m_direction(1.0f, -0.1f, 0.0f),
m_up(0, 1, 0)
{}

Camera::Camera(const glm::vec3 position):
m_eye(position),
m_direction(1.0f, -0.1f, 0.0f),
m_up(0, 1, 0)
{}

void Camera::
update_project_transform(glm::mat4 &mat) const noexcept
{
  mat = glm::perspective(glm::radians(15.0f), m_aspect_ratio, m_near_plane, m_far_plane);
}

void Camera::
update_view_transform(glm::mat4 &mat) const noexcept
{
    mat = glm::lookAt(m_eye, m_eye+m_direction, m_up);
}

void Camera::
set_fovy(float fovy)
{
  assert(fovy > 0.0 && fovy < 180.0);
  m_fovy = fovy;
}

float Camera::
get_fovy() const noexcept
{
  return m_fovy;
}

void Camera::
set_aspect_ratio(float aspect_ratio)
{
  assert(aspect_ratio > 0.0);
  m_aspect_ratio = aspect_ratio;
}

float Camera::
get_aspect_ratio() const noexcept
{
  return m_aspect_ratio;
}

float Camera::
get_near_plane() const noexcept
{
  return m_near_plane;
}

float Camera::
get_far_plane() const noexcept
{
  return m_far_plane;
}

void Camera::
set_near_plane(float near)
{
  m_near_plane = near;
}

void Camera::
set_far_plane(float far)
{
  m_far_plane = far;
}

void Camera::
set_eye(glm::vec3 eye)
{
  m_eye = eye;
}

void Camera::
set_direction(glm::vec3 direction)
{
  m_direction = glm::normalize(direction);
}

void Camera::set_up(glm::vec3 up)
{
  m_up = glm::normalize(up);
}

}
