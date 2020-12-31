set(SOURCES
    reduction.cu
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

