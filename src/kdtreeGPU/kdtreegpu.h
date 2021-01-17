#pragma once


#include "src/data/mesh.h"
#include "src/data/smallnode.h"

namespace CuKee
{
class KDTreeGPU
{
public:
  KDTreeGPU();
  ~KDTreeGPU();

  void build(); // should we take a Mesh or a DeviceMesh??

private:

  /* Large Node Stage */
  void process_large_nodes();

  /* Small Node Stage */
  void preprocess_small_nodes(); // Algo 4, PreprocessSmallnodes(smalllist: list)
  void process_small_nodes(); // Algo 5, ProcessSmallNodes

};
}
