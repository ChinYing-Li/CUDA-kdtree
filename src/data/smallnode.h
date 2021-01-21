#pragma once

#include <thrust/device_vector.h>

#include "src/data/nodelist.h"
#include "src/data/splitlist.h"

namespace CuKee
{
#define MAX_PRIM_PER_SMALL_NODE 64

namespace Device
{
struct SmallNodeList;
}


class SmallNodeList : public NodeList
{
public:
  SmallNodeList() = default;
  ~SmallNodeList() = default;

  void resize(unsigned int node_size);
  void split(); // Algo 4, PreprocessSmallNodes(smallist: list)
  Device::SmallNodeList to_device();

private:
  bool m_is_complete = false;
  thrust::device_vector<long long> m_prim_bits;
  thrust::device_vector<long long> m_small_root;  // Uppermost small node ancestor
  thrust::device_vector<unsigned int> m_root_index;
};

namespace Device
{
struct SmallNodeList
{
  long long* m_prim_bits;
  long long* m_small_root;
  unsigned int* m_root_index;
  Device::NodeList m_node_list;
};

__global__
void compute_SAH(Device::SmallNodeList activelist, Device::SplitCandidates splitcan);

__global__
void preprocess(Device::SmallNodeList activelist, Device::SplitCandidates splitcan);
}

}
