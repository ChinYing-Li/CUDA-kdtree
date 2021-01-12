// #include <assert.h>
#include <algorithm>
#include <limits>
#include <math.h>

#include "kdtreecpu.h"

namespace CuKee
{
const float infinity = std::numeric_limits<float>::max();

struct BoundEdge
{
  enum Type
  {
    START,
    END,
  };

  BoundEdge();
  BoundEdge(float t, int prim_num, bool is_start):
    m_t(t),
    m_prim_num(prim_num),
    m_type(is_start ? Type::START : Type::END)
  {}

  float m_t;
  int m_prim_num;
  Type m_type;
};

KDTreeCPU::
KDTreeCPU(const int ibo_size, const std::vector<unsigned int>& ibo,
          const int vbo_size, const std::vector<float>& vbo,
          const int intersect_cost,
          const int traversal_cost,
          const int max_depth,
          const int max_prim_per_node,
          const float empty_bonus):
  m_num_alloced_node(0),
  m_next_free_node(0),
  m_intersect_cost(intersect_cost),
  m_traversal_cost(traversal_cost),
  m_max_prim_per_node(max_prim_per_node),
  m_max_depth(max_depth <= 0 ? std::round(8 + 1.3f * log2(ibo_size / 3)) : max_depth),
  m_empty_bonus(empty_bonus),
  m_ibo_size(ibo_size),
  m_ibo(ibo),
  m_vbo_size(vbo_size),
  m_vbo(vbo)
{

}

void KDTreeCPU::build(const unsigned int node_num,
                      const AABB& node_bbox,
                      const std::vector<AABB>& prim_bbox,
                      int* prim_nums, // TODO: Check out what is prim_nums
                      const int n_prim_enclosed, // the number of primitives contained.
                      const int depth,
                      const std::unique_ptr<BoundEdge[]> edges[3],
                      int *prims0,
                      int *prims1,
                      int bad_refines)
{
  assert(node_num == m_next_free_node);

  if(m_next_free_node == m_num_alloced_node)
  {
    unsigned int new_num_alloced_node = std::max((unsigned int)512, 2 * m_num_alloced_node);
    m_nodes.reserve(new_num_alloced_node);
    m_num_alloced_node = new_num_alloced_node;
  }

  ++m_next_free_node;

  // Initialize leaf node if termination criteria is met
  if (n_prim_enclosed <= m_max_prim_per_node || depth == 0)
  {
    m_nodes[node_num].init_leaf(prim_nums, n_prim_enclosed, &m_prim_indices);
  }

  // Otherwise, initialize interior node and continue recurtion.

  // Choose split axis position for interior node
  short best_axis = -1;
  int best_offset = -1;
  float res_cost = infinity;
  float old_cost = m_intersect_cost * float(n_prim_enclosed);
  float total_SA = node_bbox.get_surface_area();
  float inv_total_SA = 1.0f / total_SA;
  glm::vec3 bbox_diagonal = node_bbox.m_max_vert - node_bbox.m_min_vert;
  // int axis = TODO: write the correct implementation
  unsigned short axis = node_bbox.get_axis_with_max_distance();
  unsigned short retries = 0;

  while (true)
  {
    for (int i = 0; i < n_prim_enclosed; ++i)
    {
      int i_prim_num = prim_nums[i];
      const AABB& bbox = (*m_prim_bbox).at(i);
      edges[axis][2 * i] = BoundEdge(bbox.m_min_vert[axis], i_prim_num, true);
      edges[axis][2 * i + 1] = BoundEdge(bbox.m_max_vert[axis], i_prim_num, false);
    }

    // Sort _edges_ for _axis_
    std::sort(&edges[axis][0], &edges[axis][2 * n_prim_enclosed],
              [](const BoundEdge &lhs, const BoundEdge &rhs) -> bool
                {
                  return (lhs.m_t == rhs.m_t)
                      ? (int)lhs.m_type < (int)rhs.m_type
                      : lhs.m_t < rhs.m_t;
                 });

    // Compute cost of all possible splits on axis to find the best split
    unsigned int n_prim_below = 0;
    unsigned int n_prim_above = n_prim_enclosed;

    for (int i = 0; i < 2 * n_prim_enclosed; ++i)
    {
      if (edges[axis][i].m_type == BoundEdge::END)
      {
        --n_prim_above;
      }

      float edge_t = edges[axis][i].m_t;
      if(edge_t > node_bbox.get_vert_element(axis, true)
         && edge_t < node_bbox.get_vert_element(axis, false))
      {
        unsigned short axis_1 = (axis + 1) % 3;
        unsigned short axis_2 = (axis + 2) % 3;

        // TODO: get the implementaion right
        float SA_below_split = (edge_t - node_bbox.get_vert_element(axis, true));
        float SA_above_split = 2.0;
        float bonus = (n_prim_above == 0 || n_prim_below == 0) ? m_empty_bonus : 0;
        float cost = m_traversal_cost +
            m_intersect_cost * (1.0 - bonus) * (SA_below_split * n_prim_below + SA_above_split * n_prim_above) * inv_total_SA;

        // Update best aplit if this is the lowest cost so far
        if (cost < res_cost)
        {
          res_cost = cost;
          best_axis = axis;
          best_offset = i;
        }
      }

      if (edges[axis][i].m_type == BoundEdge::START)
      {
        ++n_prim_below;
      }
    }

    if (best_axis == -1 && retries < 2)
    {
      ++retries;
      axis = (axis + 1) % 3;
    }
    else
    {
      break;
    }
  }

  if (res_cost > old_cost)
  {
    ++bad_refines;
  }

  if ((res_cost > 4.0 * old_cost && n_prim_enclosed < 16) || best_axis == -1 || bad_refines == 3)
  {
    m_nodes[node_num].init_leaf(prim_nums, n_prim_enclosed, &m_prim_indices);
    return;
  }

  // Now that we have determined the split position, determine whether the child
  unsigned int n0 = 0;
  unsigned int n1 = 0;

  for (int i = 0; i < best_offset; ++i)
  {
    if (edges[best_axis][i].m_type == BoundEdge::START)
    {
      prims0[n0] = edges[best_axis][i].m_prim_num;
      ++n0;
    }
  }

  for (int i = best_offset + 1; i < 2 * n_prim_enclosed; ++i)
  {
    if (edges[best_axis][i].m_type == BoundEdge::END)
    {
      prims1[n1] = edges[best_axis][i].m_prim_num;
      ++n1;
    }
  }

  // Initialize children nodes
  float t_split = edges[best_axis][best_offset].m_t;
  AABB bbox0 = node_bbox;
  AABB bbox1 = node_bbox;
  bbox0.set_vert_element(t_split, best_axis, false);
  bbox1.set_vert_element(t_split, best_axis, true);

  build(node_num + 1,
        bbox0,
        prim_bbox,
        prims0, n0,
        depth - 1,
        edges,
        prims0,
        prims1 + n_prim_enclosed,
        bad_refines);

  unsigned int child_above = m_next_free_node;
  m_nodes[node_num].init_interior(best_axis, child_above, t_split);

  build(child_above,
        bbox1,
        prim_bbox,
        prims1,
        n1,
        depth - 1,
        edges,
        prims0,
        prims1 + n_prim_enclosed,
        bad_refines);
}
}
