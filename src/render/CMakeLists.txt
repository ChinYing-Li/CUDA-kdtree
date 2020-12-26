target_include_directories(ClusterEngine
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
    ${GLFW3_INCLUDE_DIRS}
)

target_link_libraries(ClusterEngine
    PRIVATE
    OpenGL::OpenGL
    OpenGL::GLU
    ${GLEW_LIBRARIES}
    ${GLFW3_LIBRARIES}
    ASSIMP
    )
