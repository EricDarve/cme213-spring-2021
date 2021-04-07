#include <cassert>
#include <cstdio>
#include <vector>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <omp.h>

using std::cout;

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

    printf("Size of matrix: %d\n", size);
    printf("Number of threads: %d\n", n_thread);
    printf("debug flag: %s\n", debug ? "true" : "false");

    assert(n_thread >= 1);
    assert(size >= 1);

    // Setting the number of threads to use.
    // By default, openMP selects the largest possible number of threads given the processor.
    omp_set_num_threads(n_thread);

    // Input matrices A & B
    double *A = new double[size * size];
    double *B = new double[size * size];
    // Output matrix C
    double *C = new double[size * size];
    #pragma omp parallel for
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
        {
            A[i * size + j] = MatA(i, j);
            B[i * size + j] = MatB(i, j);
            C[i * size + j] = 0.;
        }

    const int BS = 8;
    assert(size % BS == 0);

    high_resolution_clock::time_point time_begin = high_resolution_clock::now();
    {
        #pragma omp parallel
        #pragma omp single
        for (int i = 0; i < size; i += BS)
        {
// Note 1: i, A, B, C are firstprivate by default
// Note 2: A, B and C are pointers
// https://www.openmp.org/wp-content/uploads/OpenMP-API-Specification-5-1.pdf#page=68
// https://www.openmp.org/wp-content/uploads/OpenMP-API-Specification-5-1.pdf#page=310
// https://www.openmp.org/wp-content/uploads/openmp-examples-5-0-1.pdf#page=103
            #pragma omp task \
            depend(in: A [i * size:BS * size], B) \
            depend(inout: C [i * size:BS * size])
            for (int ii = i; ii < i + BS; ii++)
                for (int j = 0; j < size; j++)
                    for (int k = 0; k < size; k++)
                        C[ii * size + j] = C[ii * size + j] + A[ii * size + k] * B[k * size + j];
        }
    }
    high_resolution_clock::time_point time_end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(time_end - time_begin).count();
    printf("Elapsed time [millisec]: %d\n", static_cast<int>(duration));

    if (debug) /* -g */
    {
        /* Debug: check result */
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
            {
                float d_ij = 0.;

                for (int k = 0; k < size; ++k)
                {
                    d_ij += MatA(i, k) * MatB(k, j);
                }
                assert(d_ij == C[i * size + j]);
            }

        printf("PASS\n");
    }

    return 0;
}

struct block {
    double * data;
};

void spotrf(block &A) {}
void strsm(block &A, block &B) {}
void sgemm(block &A, block &B, block &C) {}
void ssyrk(block &A, block &B) {}

void blocked_cholesky(const int NB, block **A)
{
    for (int k = 0; k < NB; k++)
    {
        #pragma omp task depend(inout: A[k][k])
        spotrf(A[k][k]);
        for (int i = k + 1; i < NB; i++)
            #pragma omp task \
            depend(in: A[k][k]) \
            depend(inout: A[k][i])
            strsm(A[k][k], A[k][i]);
        // update trailing submatrix
        for (int i = k + 1; i < NB; i++)
        {
            for (int j = k + 1; j < i; j++)
                #pragma omp task \
                depend(in: A[k][i], A[k][j]) \
                depend(inout: A[j][i])
                sgemm(A[k][i], A[k][j], A[j][i]);
            #pragma omp task depend(in: A[k][i]) depend(inout: A[i][i])
            ssyrk(A[k][i], A[i][i]);
        }
    }
}
