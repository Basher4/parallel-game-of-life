#include "lesson1.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <numeric>
#include <numbers>
#include <vector>

#include <omp.h>

const int64_t num_steps = 1e8;

double pi_st()
{
    double step = 1.0 / num_steps;
    double sum = 0.0;

    for (int i = 0; i < num_steps; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    return sum * step;
}

double pi_mt()
{
    const double step = 1.0 / num_steps;

    std::vector<double> partials(omp_get_max_threads());
    int ns = num_steps / partials.size();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double block = ns * tid;

        for (int i = 0; i < ns; i++) {
            double x = (block + i + 0.5) * step;
            partials[tid] += 4.0 / (1.0 + x * x);
        }
    }

    double sum = 0.0;
    for (auto p : partials) { sum += p; }
    return sum * step;
}

double pi_mt_reference()
{
    constexpr int NUM_THREADS = 4;

    int nthreads = NUM_THREADS;
    double pi = 0;
    double sum[NUM_THREADS] = {};
    const double step = 1.0 / num_steps;
    omp_set_num_threads(NUM_THREADS);

    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int nthrds = omp_get_num_threads();
        if (id == 0) nthreads = nthrds;

        for (int i = id; i < num_steps; i += nthrds) {
            double x = (i + 0.5) * step;
            sum[id] += 4.0 / (1.0 + x * x);
        }
    }

    for (int i = 0; i < nthreads; i++) {
        pi += sum[i];
    }

    return pi * step;
}

double pi_mt_false_sharing()
{
    constexpr int NUM_THREADS = 4;
    constexpr int PAD = 8; // Cache line size 64B.

    int nthreads = NUM_THREADS;
    double pi = 0;
    double sum[NUM_THREADS][PAD] = {};
    const double step = 1.0 / num_steps;
    omp_set_num_threads(NUM_THREADS);

#pragma omp parallel
    {
        int id = omp_get_thread_num();
        int nthrds = omp_get_num_threads();
        if (id == 0) nthreads = nthrds;

        for (int i = id; i < num_steps; i += nthrds) {
            double x = (i + 0.5) * step;
            sum[id][0] += 4.0 / (1.0 + x * x);
        }
    }

    for (int i = 0; i < nthreads; i++) {
        pi += sum[i][0];
    }

    return pi * step;
}

void lesson1()
{
    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);

    printf("Ground trugh\n%.12f\n\n", std::numbers::pi);

    double start, pi, duration;

    start = omp_get_wtime();
    pi = pi_st();
    duration = omp_get_wtime() - start;
    printf("%.12f in %.3f ms\n", pi, duration * 1e3);

    start = omp_get_wtime();
    pi = pi_mt();
    duration = omp_get_wtime() - start;
    printf("%.12f in %.3f ms\n", pi, duration * 1e3);

    start = omp_get_wtime();
    pi = pi_mt_reference();
    duration = omp_get_wtime() - start;
    printf("%.12f in %.3f ms\n", pi, duration * 1e3);

    start = omp_get_wtime();
    pi = pi_mt_false_sharing();
    duration = omp_get_wtime() - start;
    printf("%.12f in %.3f ms\n", pi, duration * 1e3);
}
