#pragma once

#include <thrust/device_vector.h>

#include "aabb.h"

namespace CuKee
{

struct DeviceArrSplitList;

/****************************************************************************************
 * Stores the data related to node-splitting
 */
class ArrSplitList
{
public:
  ArrSplitList();
  ~ArrSplitList();

  void resize();
  void split();

  DeviceArrSplitList to_device();
  void to_host(); // After some computation on device, pull the updated result back to host

private:
  bool m_on_device;
  DeviceArrSplitList* m_dev_list_ptr;

  thrust::device_vector<int> m_split_offset;
  thrust::device_vector<int> m_split_axis;
  thrust::device_vector<float> m_split_position;

};

struct DeviceArrSplitList
{

};
}
