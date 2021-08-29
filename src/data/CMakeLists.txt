set(SOURCES
        kernel/reduction.cu
        kernel/scan.cu
        kernel/utils.cu
        aabb.cpp
        chunklist.cpp
        chunklist_kernel.cu
        mesh.cpp
        mesh_aabb.cu
        nodelist.cu
        smallnode.cu
        smallnode_kernel.cu
        splitlist.cu
        )

add_library(kdtreeGPU STATIC ${SOURCES})
set_target_properties(kdtreeGPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(kdtreeGPU PROPERTIES
        COMPILE_FLAGS "${PLATFORM_COMPILE_FLAGS}"
        )

target_include_directories(kdtreeGPU
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE
        ${CUDATookit_INCLUDE_DIRS}
        )

target_link_libraries(kdtreeGPU
        PRIVATE
        ${CUDAToolkit_LIBRARY_DIR}
        )

