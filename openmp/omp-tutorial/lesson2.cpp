#include "lessons.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <numeric>
#include <numbers>
#include <vector>

#include <omp.h>

const int64_t num_steps = 1e6;

double pi_mt_critical()
{
    const double step = 1.0 / num_steps;

    double sum = 0.0;

    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();

        for (int i = 0; i < num_steps; i += nthreads) {
            double x = (i + 0.5) * step;
            double psum = 4.0 / (1.0 + x * x);

            #pragma omp critical
            sum += psum;
        }
    }

    return sum * step;
}

double pi_mt_atomic()
{
    const double step = 1.0 / num_steps;

    double sum = 0.0;

    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();

        for (int i = 0; i < num_steps; i += nthreads) {
            double x = (i + 0.5) * step;
            double psum = 4.0 / (1.0 + x * x);

            #pragma omp atomic
            sum += psum;
        }
    }

    return sum * step;
}

double pi_mt_ref_critical()
{
    const double step = 1.0 / num_steps;

    double sum = 0.0;

    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        double psum = 0.0;

        for (int i = 0; i < num_steps; i += nthreads) {
            double x = (i + 0.5) * step;
            psum += 4.0 / (1.0 + x * x);
        }

        #pragma omp critical
        sum += psum;
    }

    return sum * step;
}

double pi_mt_ref_atomic()
{
    const double step = 1.0 / num_steps;

    double sum = 0.0;

    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        double psum = 0.0;

        for (int i = 0; i < num_steps; i += nthreads) {
            double x = (i + 0.5) * step;
            psum += 4.0 / (1.0 + x * x);
        }

        #pragma omp atomic
        sum += psum;
    }

    return sum * step;
}

void lesson2()
{
    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);

    printf("Lesson 2\nGround truth: %.12f\n\n", std::numbers::pi);

    double start, pi, duration;

    start = omp_get_wtime();
    pi = pi_mt_critical();
    duration = omp_get_wtime() - start;
    printf("MT Critical:  %.12f in %.3f ms\n", pi, duration * 1e3);

    start = omp_get_wtime();
    pi = pi_mt_atomic();
    duration = omp_get_wtime() - start;
    printf("MT Atomic:    %.12f in %.3f ms\n", pi, duration * 1e3);

    start = omp_get_wtime();
    pi = pi_mt_ref_critical();
    duration = omp_get_wtime() - start;
    printf("Ref Critical: %.12f in %.3f ms\n", pi, duration * 1e3);

    start = omp_get_wtime();
    pi = pi_mt_ref_atomic();
    duration = omp_get_wtime() - start;
    printf("Ref Atomic:   %.12f in %.3f ms\n", pi, duration * 1e3);

    printf("\n\n");
}