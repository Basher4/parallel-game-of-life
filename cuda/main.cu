#include <chrono>
#include <cstdio>
#include <omp.h>

#include <cuda_runtime.h>
#include <curand.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "conway.cuh"

const u64 GRID_SIZE = 1 << 10;
const u64 SIM_STEPS = 1 << 10;
const u64 SEED = 0xDEAD'BEEF;

void fill_buffer_with_random_data(thrust::device_vector<u8>& buf, u64 seed, bool normalize);

void naive();

int main()
{
    naive();

    return 0;
}

void naive()
{
    printf("Simulating %I64d steps of Game of Life on a %I64d x %I64d grid\n", SIM_STEPS, GRID_SIZE, GRID_SIZE);

    thrust::device_vector<u8> dev_a(GRID_SIZE * GRID_SIZE), dev_b(GRID_SIZE * GRID_SIZE);
    conway::fill<<<GRID_SIZE / 1024, 1024>>>(0, dev_a.data().get(), dev_a.size());
    fill_buffer_with_random_data(dev_a, SEED, true);
    cudaDeviceSynchronize();

    auto begin = std::chrono::high_resolution_clock::now();

    dim3 block(256, 1, 1);
    dim3 grid(GRID_SIZE / block.x, GRID_SIZE / block.y, 1);

    for (int i = 0; i < SIM_STEPS / 2; i++)
    {
        conway::naive<<<grid, block>>>(dev_a.data().get(), dev_b.data().get(), GRID_SIZE, GRID_SIZE);
        conway::naive<<<grid, block>>>(dev_b.data().get(), dev_a.data().get(), GRID_SIZE, GRID_SIZE);
    }
    CUDA_CALL(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    double time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    printf("Time to run the simulation: %.3fus\n", time_us);
    printf("Time per step: %.3fus\n", time_us / SIM_STEPS);
}


__global__ void normalize_kernel(u8* data, u64 len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len) return;
    data[tid] = data[tid] > 127 ? 1 : 0;
}

void fill_buffer_with_random_data(thrust::device_vector<u8>& buf, u64 seed, bool normalize)
{
    assert(buf.size() % 4 == 0);

    curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CURAND_CALL(curandGenerate(gen, (u32*)buf.data().get(), buf.size() / 4));

    if (normalize)
    {
        int block = 512;
        int grid = (buf.size() + block - 1) / block;
        normalize_kernel<<<grid, block>>>(buf.data().get(), buf.size());
        CUDA_CALL(cudaDeviceSynchronize());
    }
}
