#pragma once

#include <thrust/device_vector.h>

#include "aabb.h"

namespace CuKee
{

struct DeviceSplitRec;

/****************************************************************************************
 * Stores the data related to node-splitting
 */
class SplitRec
{
public:
  SplitRec();
  ~SplitRec();

  void resize();
  void split();

  DeviceSplitRec to_device();

private:
  thrust::device_vector<int> m_split_offset;
  thrust::device_vector<int> m_split_axis;
  thrust::device_vector<float> m_split_position;

};

struct DeviceSplitRec
{

};
}
