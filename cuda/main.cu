#include <chrono>
#include <cstdio>
#include <omp.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "conway.cuh"

const u64 GRID_SIZE = 1 << 10;
const u64 SIM_STEPS = 1 << 10;

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

    auto begin = std::chrono::high_resolution_clock::now();

    dim3 block(256, 1, 1);
    dim3 grid(GRID_SIZE / block.x, GRID_SIZE / block.y, 1);

    for (int i = 0; i < SIM_STEPS / 2; i++)
    {
        conway::naive << <grid, block >> > (dev_a.data().get(), dev_b.data().get(), GRID_SIZE, GRID_SIZE);
        conway::naive << <grid, block >> > (dev_b.data().get(), dev_a.data().get(), GRID_SIZE, GRID_SIZE);
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    double time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    printf("Time to run the simulation: %.3fus\n", time_us);
    printf("Time per step: %.3fus\n", time_us / SIM_STEPS);
}
