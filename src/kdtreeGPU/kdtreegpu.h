#pragma once

#include <thrust/device_vector.h>

#include "src/data/mesh.h"
#include "src/data/chunklist.h"
#include "src/data/smallnode.h"

namespace CuKee
{
class KDTreeGPU
{
public:
  KDTreeGPU(int small_thres = 64);
  ~KDTreeGPU() = default;

  void build(); // should we take a Mesh or a DeviceMesh??

private:
  int m_small_threshold;
  ChunkList m_active_list;
  ChunkList m_next_list;
  SmallNodeList m_small_list;
  Mesh m_mesh;

  thrust::device_vector<int> m_small_node_flag;

  /* Large Node Stage */
  void process_large_nodes();
  void filter_small_nodes(ChunkList* nextlist);
  void split_large_nodes();


  /* Small Node Stage */
  void preprocess_small_nodes(); // Algo 4, PreprocessSmallnodes(smalllist: list)
  void process_small_nodes(); // Algo 5, ProcessSmallNodes


};
}
