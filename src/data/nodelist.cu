#include "nodelist.h"

namespace CuKee
{
NodeList::NodeList()
{

}

inline void NodeList::
clear()
{
  m_num_node = m_num_prim = 0;
  m_left_child_indices.clear();
  m_right_child_indices.clear();
  m_starting_indices_in_prim.clear();
  m_node_prim_num.clear();
  m_prim_indices.clear();
  m_splitlist.clear();
}

inline void NodeList::
resize_node(unsigned int size)
{
  m_num_node = size;
  m_left_child_indices.resize(size);
  m_right_child_indices.resize(size);
  m_starting_indices_in_prim.resize(size);
  m_node_prim_num.resize(size);
  m_splitlist.resize(size);
}

inline void NodeList::
resize_prim(unsigned int size)
{
  m_num_prim = size;
  m_prim_indices.resize(size);
}

/****************************************************************************************
 * Remove the empty space in large nodes
 */

__global__
void cut_empty_space()
{

}

void NodeList::
cut_empty_space()
{

}

/*
 * Split large nodes into child nodes
 */
void NodeList::
split_node()
{

}

inline DeviceNodeList NodeList::
to_device()
{

}


}
