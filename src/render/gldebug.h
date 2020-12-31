#pragma once

#include "glincludes.h"

namespace CuKee
{
struct glsl_type_set
{
  GLenum      type;
  const char* name;
};

void glsl_print_uniforms(unsigned int program); // We should let the shader print their own uniforms!!
void gl_debug();
}
