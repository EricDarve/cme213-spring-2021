#include <cassert>
#include <cstdio>
#include <vector>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <cstdlib>

#include <omp.h>

using std::abs;
using namespace std::chrono;
using std::max;
using std::vector;

void ProcessOpt(int argc, char **argv, int *size, int *n_thread, bool *debug)
{
    int c;
    *size = 512;
    *n_thread = 2;
    *debug = true;

    while ((c = getopt(argc, argv, "n:p:gh")) != -1)
        switch (c)
        {
        case 'n':
            *size = atoi(optarg);
            break;

        case 'p':
            *n_thread = atoi(optarg);
            break;

        case 'g':
            *debug = true;
            break;

        case 'h':
            printf(
                "Options:\n-n SIZE\t\tMatrix size\n-p NTHREAD\tNumber of threads\n");
            exit(1);

        case '?':
            break;
        }
}

/* A(i,j) */
float MatA(int i, int j)
{
    if (i % 2)
    {
        return 1;
    }

    return -1;
}

/* B(i,j) */
float MatB(int i, int j)
{
    if ((i + j) % 2)
    {
        return i;
    }

    return j;
}

int main(int argc, char **argv)
{
    // Command line options
    int size, n_thread;
    bool debug;
    ProcessOpt(argc, argv, &size, &n_thread, &debug);

    assert(n_thread >= 1);
    assert(size >= 1);
    printf("Size of matrix = %d\n", size);
    printf("Number of threads to create = %d\n", n_thread);

    // Setting the number of threads to use.
    // By default, openMP selects the largest possible number of threads given the processor.
    omp_set_num_threads(n_thread);

    // Output matrix C
    vector<float> mat_c(size * size);

    high_resolution_clock::time_point time_begin = high_resolution_clock::now();
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
        {
            float c_ij = 0;
            for (int k = 0; k < size; ++k)
            {
                c_ij += MatA(i, k) * MatB(k, j);
            }
            mat_c[i * size + j] = c_ij;
        }
    high_resolution_clock::time_point time_end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(time_end - time_begin).count();
    printf("Elapsed time [millisec]: %d\n", static_cast<int>(duration));

    if (debug) /* -g */
    {
        /* Debug: check result */
        vector<float> mat_d(size * size);

        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
            {
                float d_ij = 0.;

                for (int k = 0; k < size; ++k)
                {
                    d_ij += MatA(i, k) * MatB(k, j);
                }
                assert(d_ij == mat_c[i * size + j]);
            }

        printf("PASS\n");
    }

    return 0;
}
