#include "conway.cuh"

__device__ __host__ bool should_cell_live(u8 is_alive, i8 neighbors)
{
	if (neighbors <= 1) return false;
	if (neighbors == 2) return true;
	if (neighbors == 3) return is_alive;
	if (neighbors >= 4) return false;
}

__global__ void conway::naive(u8* src_data, u8* dst_data, u64 size_x, u64 size_y)
{
	auto src = [=](i64 x, i64 y) { return src_data[y * size_x + x]; };
	auto dst = [=](i64 x, i64 y) { return &dst_data[y * size_x + x]; };
	auto in_range = [=](i64 x, i64 y) { return x >= 0 && x < size_x && y >= 0 && y <= size_y; };

	i64 ix = blockIdx.x * blockDim.x + threadIdx.x;
	i64 iy = blockIdx.y * blockDim.y + threadIdx.y;
	if (ix >= size_x || iy >= size_y) return;

	i8 sum = 0;
	for (i8 dy = -1; dy <= 1; dy++) {
		for (i8 dx = -1; dx <= 1; dx++) {
			i64 x = ix + dx;
			i64 y = iy + dy;

			if (in_range(x, y)) {
				sum += src(x, y);
			}
		}
	}

	// Writing data only if state changed makes no difference to runtime in this case.
	*dst(ix, iy) = should_cell_live(src(ix, iy), sum) ? 1 : 0;
}

__global__ void conway::fill(u8 value, u8* data, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= len) return;

	data[tid] = value;
}
