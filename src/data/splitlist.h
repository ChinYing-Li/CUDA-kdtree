#pragma once

#include <thrust/device_vector.h>

#include "aabb.h"

namespace CuKee
{
class ChunkList;

namespace Device {
struct SplitData
{
  int* m_split_offset;
  int* m_split_size;
  int* m_split_axis;
  float* m_split_position;

  void clear();

  void resize(unsigned int size)
  {
    // TODO: resize these members !
  }
};

struct SplitCandidates
{
  int* m_prim_indices;
  int* m_starting_indices_in_prim; // indexing into array of primitives
  unsigned long long* m_left_prim_bitmask;
  unsigned long long* m_right_prim_bitmask;
  int* m_root_index;
  Device::SplitData m_split_data;
};
}

/****************************************************************************************
 * Stores the data related to node-splitting
 */

struct SplitData
{
  thrust::device_vector<int> m_split_first_indices; //
  thrust::device_vector<int> m_split_size;
  thrust::device_vector<int> m_split_axis;
  thrust::device_vector<float> m_split_position;

  void clear();
  void resize(unsigned int size);

  Device::SplitData to_device()
  {
    Device::SplitData split_data;
    split_data.m_split_offset = thrust::raw_pointer_cast(&m_split_first_indices[0]);
    split_data.m_split_size = thrust::raw_pointer_cast(&m_split_size[0]);
    split_data.m_split_axis = thrust::raw_pointer_cast(&m_split_axis[0]);
    split_data.m_split_position = thrust::raw_pointer_cast(&m_split_position[0]);
    return split_data;
  }
};

class SplitCandidates
{
public:
  SplitCandidates() = default;
  ~SplitCandidates() = default;

  void resize(unsigned int size);
  void clear();
  void split(ChunkList& list);

  Device::SplitCandidates to_device();
  void to_host(); // After some computation on device, pull the updated result back to host

protected:
  bool m_on_device;
  Device::SplitCandidates* m_dev_list_ptr;

  SplitData m_split_data;
  thrust::device_vector<int> m_prim_indices;
  thrust::device_vector<int> m_starting_indices_in_prim; // indexing into array of primitives
  thrust::device_vector<unsigned long long> m_left_prim_bitmask;
  thrust::device_vector<unsigned long long> m_right_prim_bitmask;
  thrust::device_vector<int> m_root_index; // What's the usage of this member?
  // first_split_index -> What is this?
};

enum SplitSide
{
  LEFT = -1,
  RIGHT = 1,
};
}
