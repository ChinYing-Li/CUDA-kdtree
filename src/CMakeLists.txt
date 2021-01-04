set(SOURCES
    kdtreeCPU/
    render/camera.cpp
    render/window.cpp
    utils/timer.cpp
    aabb.cpp
    kdtree.cpp
    mesh.cpp
    )

add_library(CUDA_KDTREE ${SOURCES})
add_library(STB_IMAGE "ext/stb_image.cpp")
add_library(OBJLOADER "ext/objloader.cpp")

target_include_directories(CUDA_KDtree
    PUBLIC
    ${EXT_SOURCE_DIRS}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDATookit_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
    ${GLFW3_INCLUDE_DIRS}
)

target_link_libraries(CUDA_KDtree
    PRIVATE
    ${CUDAToolkit_LIBRARY_DIR}
    OpenGL::OpenGL
    OpenGL::GLU
    ${GLEW_LIBRARIES}
    ${GLFW3_LIBRARIES}
    ASSIMP
    ${STB_IMAGE}
    ${OBJLOADER}
    )

install(TARGETS CUDA_KDtree
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib)

# Add main.cpp
add_executable(CuKee-demo src/main.cpp)
target_link_libraries(Cukee-demo
                      CUDA_KDtree
                      )


