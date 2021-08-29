// CUDA Dependencies
#include <cuda.h>

namespace CuKee
{
    template<typename T, class ReductionOp, unsigned int block_size>
    __inline__ __device__
    T reduction_device(T* arr);
}
