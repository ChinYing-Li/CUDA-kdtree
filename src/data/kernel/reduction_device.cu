#include "reduction_device.h"

namespace CuKee
{
    template<typename T, class ReductionOp, unsigned int block_size>
    __inline__ __device__
    T reduction_device(T* arr)
    {
        unsigned int thread = threadIdx.x;
        if(block_size >= 512)
        {
            if(thread < 256)
            {
                arr[thread] = ReductionOp::op(arr[thread], arr[thread + 256]);
            }
            __syncthreads();
        }

        if(block_size >= 256)
        {
            if(thread < 128)
            {
                arr[thread] = ReductionOp::op(arr[thread], arr[thread + 128]);
            }
        }
        __syncthreads();

        if(block_size >= 128)
        {
            if(thread < 64)
            {
                arr[thread] = ReductionOp::op(arr[thread], arr[thread + 64]);
            }
        }
        __syncthreads();

        if(block_size >= 64)
        {
            if(thread < 32)
            {
                arr[thread] = ReductionOp::op(arr[thread], arr[thread + 32]);
            }
        }
        __syncthreads();

        if(block_size >= 32)
        {
            if(thread < 16)
            {
                arr[thread] = ReductionOp::op(arr[thread], arr[thread + 16]);
            }
        }
        __syncthreads();

        if(block_size >= 16)
        {
            if(thread < 8)
            {
                arr[thread] = ReductionOp::op(arr[thread], arr[thread + 8]);
            }
        }
        __syncthreads();

        if(block_size >= 8)
        {
            if(thread < 4)
            {
                arr[thread] = ReductionOp::op(arr[thread], arr[thread + 4]);
            }
        }
        __syncthreads();

        if(block_size >= 4)
        {
            if(thread < 2)
            {
                arr[thread] = ReductionOp::op(arr[thread], arr[thread + 2]);
            }
        }
        __syncthreads();

        if(block_size >= 2)
        {
            if(thread < 1)
            {
                arr[thread] = ReductionOp::op(arr[thread], arr[thread + 1]);
            }
        }
        __syncthreads();

        return arr[0];
    }

}


