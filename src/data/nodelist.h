#pragma once

#include <thrust/device_vector.h>

#include "src/data/splitlist.h"

namespace CuKee
{

struct DeviceNodeList
{
  int* m_left_child_indices;
  int* m_right_child_indices;
  int* m_prim_starting_indices;
  int* m_node_prim_num;
  int* m_prim_indices;
  DeviceSplitData m_split_data;
  int* m_num_node;
  int* m_num_prim;
};

class NodeList
{
public:
  NodeList();
  ~NodeList();
  virtual void clear();
  virtual void resize_node(unsigned int size);
  virtual void resize_prim(unsigned int size);
  unsigned int num_node();
  unsigned int num_prim();
  void cut_empty_space();
  void split_node();

  DeviceNodeList to_device();

protected:
  unsigned int m_num_node;
  unsigned int m_num_prim;

  thrust::device_vector<int> m_left_child_indices;
  thrust::device_vector<int> m_right_child_indices;
  thrust::device_vector<int> m_prim_indices;
  thrust::device_vector<int> m_starting_indices_in_prim;
  thrust::device_vector<int> m_node_prim_num;
  thrust::device_vector<int> m_node_depth;

  SplitData m_splitdata;

  void update_starting_indices();
};
}
