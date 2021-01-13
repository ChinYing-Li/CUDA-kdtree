#pragma once

#include "vector_type.h"

__device__
float3 float3_minimum_op::
operator()(const float3& lhs, const float3& rhs)
{
  return make_float3(fminf(lhs.x, rhs.x),
                     fminf(lhs.y, rhs.y),
                     fminf(lhs.z, rhs.z));
}

__device__
float3 float3_maximum_op::
operator()(const float3& lhs, const float3& rhs)
{
  return make_float3(fmaxf(lhs.x, rhs.x),
                     fmaxf(lhs.y, rhs.y),
                     fmaxf(lhs.z, rhs.z));
}

__device__
float4 float4_minimum_op::operator()(const float4& lhs, const float4& rhs)
{
	return make_float4(fminf(lhs.x, rhs.x),
	                     fminf(lhs.y, rhs.y),
	                     fminf(lhs.z, rhs.z),
						 fminf(lhs.w, rhs.w));
}

__device__
float4 float4_maximum_op::operator()(const float4& lhs, const float4& rhs)
{
	return make_float4(fminf(lhs.x, rhs.x),
	                     fminf(lhs.y, rhs.y),
	                     fminf(lhs.z, rhs.z),
						 fminf(lhs.w, rhs.w));
}
