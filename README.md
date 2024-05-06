# Parallel Implementations of Conway's Game Of Life

Since there is virtually zero computation, this is an exercise in how to optimise memory access for parallel algorithms. I intend to implement the same set of features using:

## CUDA
1. Naive implementation
    - Every cell takes up one byte
    - No attempt to cache data
1. One thread processes multiple cells
1. Use 1 bit to represent a cell
1. Use local caches
    - Memory-wise this problem is not very different from matrix multiplication
1. Use CUDA - DX12/Vulkan/OpenGL interop to visualise the algorithm

## OpenMP
I found [OpenMP lectures on youtube](https://www.youtube.com/playlist?list=PLLX-Q6B8xqZ8n8bwjGdzBJ25X2utwnoEG). We'll see what comes out of it.

## Vulkan + (Compute) Shaders
Most basic example to learn how to set up a Vulkan app.
