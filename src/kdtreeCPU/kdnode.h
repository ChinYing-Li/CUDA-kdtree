#pragma once

#include <vector>

namespace CuKee
{
/*
 *
 */
struct KDNode
{
public:
  void init_leaf(int *primitive_counts, int num_primitive, std::vector<int> *primitive_indices);
  void init_interior(int axis, int ac, float split);

  bool is_leaf() const;
  int get_child_above() const;
  int get_num_primitive() const;
  int get_split_axis() const;
  float get_split_position() const;

  union
  {
    float m_split;
    int m_one_primitive;
    int m_primitive_indices_offset;
  }; // 4 bytes

private:

  union
  {
    int m_flags; /* Parameter used by both kinds of KDNode */
    int m_num_primitive;
    int m_child_above;
  }; // 4 bytes
};

}

