#include "kdnode.h"

namespace CuKee
{
/*
 *
 */
void KDNode::
init_leaf(int *primitive_counts, int num_primitive, std::vector<int> *primitive_indices)
{
  m_flags = 3;
  m_num_primitive |= (num_primitive << 2);

  if (num_primitive == 0)
  {
    m_one_primitive = 0;
  }
  else if (num_primitive == 1)
  {
    m_one_primitive = primitive_counts[0];
  }
  else
  {

  }
}

void KDNode::
init_interior(int axis, int ac, float split)
{
  m_split = split;
  m_flags = axis;
  m_child_above |= (ac << 2);
}

bool KDNode::
is_leaf() const
{
  return (m_flags & 3) == 3;
}


int KDNode::
get_child_above() const
{
  return m_child_above >> 2;
}

int KDNode::
get_num_primitive() const
{
  return m_num_primitive >> 2;
}

int KDNode::
get_split_axis() const
{
  return m_flags & 3;
}

float KDNode::
get_split_position() const
{

}
}
