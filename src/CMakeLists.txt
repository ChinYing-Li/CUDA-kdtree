set(SOURCES
        data/kernel/reduction.cu
        data/kernel/reduction_device.cu
        data/kernel/scan.cu
        data/kernel/utils.cu
        data/aabb.cpp
        data/chunklist.cpp
        data/chunklist_kernel.cu
        data/mesh.cpp
        data/mesh_aabb.cu
        data/nodelist.cu
        data/smallnode.cu
        data/smallnode_kernel.cu
        data/splitlist.cu
        kdtreeCPU/kdnode.cpp
        kdtreeCPU/kdtreecpu.cpp
        kdtreeGPU/kdtreegpu.cu
        math/vector_type.cu
        render/camera.cpp
        render/window.cpp
        utils/timer.cpp
        kdtree.cpp
    )

add_library(CUDA_KDTree ${SOURCES})

if(UNIX AND NOT APPLE)
    set(CUDA_SAMPLE_INCLUDE_DIRS "/usr/local/cuda/samples/common")
endif()

# Configure Thrust
# thrust_create_target(Thrust)

target_include_directories(CUDA_KDTree
    PUBLIC
    ${EXT_SOURCE_DIRS}
    PRIVATE
        ${PROJECT_SOURCE_DIR}/ext
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/data
        ${CMAKE_CURRENT_SOURCE_DIR}/data/kernel
        ${CMAKE_CURRENT_SOURCE_DIR}/kdtreeCPU
        ${CMAKE_CURRENT_SOURCE_DIR}/kdtreeGPU
        ${CMAKE_CURRENT_SOURCE_DIR}/math
        ${CMAKE_CURRENT_SOURCE_DIR}/render
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
        ${CUDATookit_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIR}
        ${GLEW_INCLUDE_DIRS}
        ${GLFW3_INCLUDE_DIRS}
        ${BOOST_INCLUDE_DIRS}
        ${CUDA_SAMPLE_INCLUDE_DIRS}
)

target_link_libraries(CUDA_KDTree
    PRIVATE
    ${CUDAToolkit_LIBRARY_DIR}
    OpenGL::OpenGL
    OpenGL::GLU
    ${GLEW_LIBRARIES}
    ${GLFW3_LIBRARIES}
    ${BOOST_LIBRARIES}
    ASSIMP
    ${OBJLOADER}
    Thrust
    )

set_target_properties(CUDA_KDTree
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

list(APPEND CUDA_NVCC_FLAG -Xcompiler)

install(TARGETS CUDA_KDTree
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib)

# Add main.cpp
add_executable(CuKee_demo application.cpp)
target_link_libraries(CuKee_demo
                      CUDA_KDTree
                      )

if(APPLE)
    set_property(TARGET CuKee_demo
        PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
