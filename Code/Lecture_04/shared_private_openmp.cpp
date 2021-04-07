#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#define UNITTIME 1000

int main(int argc, char **argv)
{

    /* Command line option */
    const int n_thread = 4;

    assert(n_thread >= 1);

    // Set number of threads for this example
    // Not needed if you want to use the default number of threads
    omp_set_num_threads(n_thread);

    int shared_int = -1;

#pragma omp parallel
    {
        // shared_int is shared
        int tid = omp_get_thread_num();
        printf("Thread ID %2d | shared_int = %d\n", tid, shared_int);
    }

    int is_private = -2;

#pragma omp parallel private(is_private)
    {
        int tid = omp_get_thread_num();
        int rand_tid = rand();
        is_private = rand_tid;
        printf("Thread ID %2d | is_private = %d\n", tid, is_private);
        assert(is_private == rand_tid);
    }

    printf("Main thread | is_private = %d\n", is_private);

    return 0;
}
