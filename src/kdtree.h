#pragma once

#include <memory>

// CUDA Dependencies
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>

namespace CuKee
{
class KDTreeNode;
class KDTree
{
public:
  KDTree();
  KDTree(float3* triangles);
  virtual void build_tree() = 0;
  virtual KDTreeNode *get_root_node() = 0;

private:
  int m_num_vertices;
  int m_num_triangles;
  KDTreeNode* m_root;
};
}
