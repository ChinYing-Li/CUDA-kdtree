#include <iostream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "window.h"

namespace CuKee
{
Window::
Window()
{

};

Window::
~Window()
{
  quit();
}

void Window
::quit()
{
    glfwDestroyWindow(m_window_ptr.get());
    glfwTerminate();
    exit(EXIT_SUCCESS);
}

void Window
::init_glfw()
{
    glfwSetErrorCallback(error_callback);
    if(!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW library." << std::endl;
        exit(-1);
    }
}

void Window
::init_glew()
{
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "Error: Failed to initialise GLEW : "
                  << glewGetErrorString(err)
                  << std::endl;
        exit (-1);
    }
}

void Window::
display_gl_info()
{
    std::cout << "VENDOR:   " << glGetString(GL_VENDOR) << std::endl;
    std::cout << "RENDERER: " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "VERSION:  " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL:     " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    return;
}
}

