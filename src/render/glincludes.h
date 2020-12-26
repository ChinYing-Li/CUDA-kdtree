#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "debug.h"

struct color_t
{
    int r;
    int g;
    int b;
};

const color_t COLOR_RED = { 236, 100, 75 };
const color_t COLOR_GREEN = { 135, 211, 124 };
const color_t COLOR_BLACK = { 52, 73, 94 };
const color_t COLOR_BACKGROUND = { 242, 241, 239 };

class GLMatrices
{
public:
    GLMatrices(){};
    ~GLMatrices() = default;
    
    glm::mat4 projection;
    glm::mat4 model;
    glm::mat4 view;
    GLuint    MatrixID;
    GLuint    Tr;
};
