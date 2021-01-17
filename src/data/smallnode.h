#pragma once

#include <cuda_runtime.h>

#include <thrust/device_vector.h>

#include "src/data/splitlist.h"

namespace CuKee
{
struct DeviceArrSmallNode;

class ArrSmallNode
{
public:
  ArrSmallNode();
  ~ArrSmallNode();

  void get_best_split(); // Algo 4, PreprocessSmallNodes(smallist: list)
  DeviceArrSmallNode to_device();

private:
  thrust::device_vector<long long> m_prim_bits;
  thrust::device_vector<long long> m_small_root;
  thrust::device_vector<unsigned int> m_root_index;
};

struct DeviceArrSmallNode
{

};

__global__ void cu_compute_SAH();
}
