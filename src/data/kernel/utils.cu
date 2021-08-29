#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <boost/shared_ptr.hpp>

// #include <helper_math.h>

#include "math/vector_type.h"

namespace CuKee
{
    struct CountBitsFunctor:
    public thrust::unary_function<int, int>
    {
        template<typename T> __device__ __host__
        int operator()(T val)
        {
            unsigned int count;
            for(count = 0; val; ++count)
            {
                val &= val - 1;
            }
            return count;
        }
    };

    struct FillLowestBitsFunctor:
    public thrust::unary_function<int, int>
    {
        __device__
        unsigned long long operator()(int val)
        {
            unsigned long long res = (unsigned long long) 1 << val;
            return res - 1;
        }
    };

    template<typename T>
    struct IsNoneZero
    {
        __host__ __device__ __inline__
        bool operator()(T val)
        {
            return val != 0;
        }
    };


}