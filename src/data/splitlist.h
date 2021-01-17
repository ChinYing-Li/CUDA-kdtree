#pragma once

#include <thrust/device_vector.h>

#include "aabb.h"

namespace CuKee
{

struct DeviceSplitList
{

};

/****************************************************************************************
 * Stores the data related to node-splitting
 */
class SplitList
{
public:
  SplitList();
  ~SplitList();

  void resize(unsigned int size);
  void clear();
  void split();

  DeviceSplitList to_device();
  void to_host(); // After some computation on device, pull the updated result back to host

protected:
  bool m_on_device;
  DeviceSplitList* m_dev_list_ptr;

  thrust::device_vector<int> m_split_offset;
  thrust::device_vector<int> m_split_axis;
  thrust::device_vector<float> m_split_position;

};


}
