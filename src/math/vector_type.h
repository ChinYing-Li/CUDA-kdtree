#pragma once

#include <cuda_runtime.h>

struct float3_minimum_op
{
  __device__
  float3 operator()(const float3& lhs, const float3& rhs);
};

struct float3_maximum_op
{
  __device__
  float3 operator()(const float3& lhs, const float3& rhs);
};

struct float4_minimum_op
{
	__device__
	float4 operator()(const float4& lhs, const float4& rhs);
};

struct float4_maximum_op
{
	__device__
	float4 operator()(const float4& lhs, const float4& rhs);
};
