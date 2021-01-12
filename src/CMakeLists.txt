set(SOURCES
    kdtreeCPU/kdnode.cpp
    kdtreeCPU/kdnode.h
    kdtreeCPU/kdtreecpu.cpp
    kdtreeCPU/kdtreecpu.h
    math/vector_type.cuh
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
    ${BOOST_INCLUDE_DIRS}
)

target_link_libraries(CUDA_KDtree
    PRIVATE
    ${CUDAToolkit_LIBRARY_DIR}
    OpenGL::OpenGL
    OpenGL::GLU
    ${GLEW_LIBRARIES}
    ${GLFW3_LIBRARIES}
    ${BOOST_LIBRARIES}
    ASSIMP
    ${STB_IMAGE}
    ${OBJLOADER}
    )

set_target_properties(TARGET CUDA_kdtree
    PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS CUDA_KDtree
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib)

# Add main.cpp
add_executable(CuKee_demo src/application.cpp)
target_link_libraries(Cukee_demo
                      CUDA_KDtree
                      )

if(APPLE)
    set_property(TARGET CuKee_demo
        PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
