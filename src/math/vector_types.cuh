#pragma once


template <typename T>
struct minimum
{
  __device__
  T operator()(const T& lhs, const T& rhs);
};
