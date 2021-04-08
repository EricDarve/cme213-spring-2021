#include <omp.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <vector>
#include <cassert>

using namespace std;

#include "randomMT.h"

float force(const float x)
{
    return -2. * atan(x) / (x * x + 1.);
}

int main(void)
{
    const int n = 64;
    vector<float> x(n);

    /* Generate random points on the unit interval */
    #pragma omp parallel
    {
        const long tid = omp_get_thread_num();
        randMT r(4357U + unsigned(tid));
        /* Thread safe generation of random numbers */
        #pragma omp for
        for (int i = 0; i < n; ++i)
            x[i] = r.rand();
    }

    /* Compute interaction forces between particles.
     Atomic is used. */
    vector<float> f(n);
    #pragma omp parallel for
    for (int i = 0; i < n; ++i)
        f[i] = 0.;

    #pragma omp parallel for
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
        {
            const float x_ = x[i] - x[j];
            const float f_ = force(x_);
            #pragma omp atomic
            f[i] += f_;
            #pragma omp atomic
            f[j] -= f_;
        }

    // Test
    {
        vector<float> f0(n, 0.);

        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
            {
                const float x_ = x[i] - x[j];
                const float f_ = force(x_);
                f0[i] += f_;
                f0[j] -= f_;
            }

        for (int i = 0; i < n; ++i)
            assert(fabs(f0[i] - f[i]) < 1e-4);

        float max = 0;
        for (int i = 0; i < n; ++i)
            max = max < fabs(f0[i] - f[i]) ? fabs(f0[i] - f[i]) : max;
        printf("largest error %g\n", max);

        cout << "PASS\n";
    }

    return 0;
}
