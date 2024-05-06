#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "types.cuh"

namespace conway {
	/// <summary>
	/// Dummy function to test if cuda actually works.
	/// </summary>
	__global__ void fill(u8 value, u8* data, int len);

	/// <summary>
	/// Stupid implementation.
	/// </summary>
	__global__ void naive(u8* src, u8* dst, u64 size_x, u64 size_y);
}