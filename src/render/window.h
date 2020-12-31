#pragma once

#include <memory>

class GLFWwindow;

namespace CuKee
{
class Window
{
public:
  Window();
  Window(const unsigned int width, const unsigned int height);
  ~Window();
  void quit();

  static void error_callback(int error, const char *description);
  static void display_gl_info();

  std::unique_ptr<GLFWwindow> m_window_ptr;

private:
  void init_glfw();
  void init_glew();
};
}
