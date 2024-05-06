#pragma once

#include <cuda/std/cstdint>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return;}} while(0)

using i8 = cuda::std::int8_t;
using u8 = cuda::std::uint8_t;
using i16 = cuda::std::int16_t;
using u16 = cuda::std::uint16_t;
using i32 = cuda::std::int32_t;
using u32 = cuda::std::uint32_t;
using i64 = cuda::std::int64_t;
using u64 = cuda::std::uint64_t;
