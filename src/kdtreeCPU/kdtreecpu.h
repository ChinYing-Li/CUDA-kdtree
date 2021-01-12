#pragma once

#include <memory>
#include <vector>

#include "aabb.h"
#include "alignedvector.h"
#include "kdnode.h"

namespace CuKee
{
struct BoundEdge;

/*
 * CPU-based KD Tree implementation using greedy surface area heuristic.
 * The algorithm is based on "",
 * and the code references that of "Physically-Based Rendering Book"
 */
class KDTreeCPU
{
public:
  KDTreeCPU() = delete ;
  KDTreeCPU(const int ibo_size, const std::vector<unsigned int>& ibo,
            const int vbo_size, const std::vector<float>& vbo,
            const int intersect_cost = 80,
            const int traversal_cost = 1,
            const int max_depth = -1,
            const int max_prim_per_node = 1,
            const float empty_bonus = 0.5);
  ~KDTreeCPU();


private:
  unsigned int m_num_alloced_node;
  unsigned int m_next_free_node;
  int m_intersect_cost;
  int m_traversal_cost;
  int m_max_prim_per_node;
  int m_max_depth;
  float m_empty_bonus;
  AABB m_bbox;

  int m_ibo_size;
  std::vector<unsigned int> m_ibo;
  int m_vbo_size;
  std::vector<float> m_vbo;

  std::vector<int> m_prim_indices;
  std::vector<AABB> *m_prim_bbox;
  aligned_vector<KDNode, 8> m_nodes;

  void build(const unsigned int node_num,
             const AABB& node_bbox,
             const std::vector<AABB>& prim_bbox,
             int* prim_nums, // TODO: Check out what is prim_nums
             const int n_primitives,
             const int depth,
             const std::unique_ptr<BoundEdge[]> edges[3],
             int *prims0,
             int *prims1,
             int bad_refines);
};

struct KDTask
{
  const KDNode* node;
  float m_tmin;
  float m_tmax;
};
}
