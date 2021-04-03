#include <omp.h>
#include <cstdio>
#include <vector>
#include <sstream>
#include <iomanip>
#include <chrono>

using namespace std;

#define SCALE 10000
#define ARRINIT 2000

void DoWork(long tid, int &ndigits, long &etime)
{
    auto start = chrono::system_clock::now();

    const int digits = (2 + tid) * 28;
    ndigits = (2 + tid) * 8;

    ostringstream pi;
    pi.fill('0');
    std::vector<long> arr(digits + 1);

    for (int i = 0; i <= digits; ++i)
        arr[i] = ARRINIT;

    long sum = 0;

    for (int i = digits; i > 0; i -= 14)
    {
        int carry = sum % SCALE;
        sum = 0;

        for (int j = i; j > 0; --j)
        {
            sum = sum * j + SCALE * arr[j];
            arr[j] = sum % ((j << 1) - 1);
            sum /= (j << 1) - 1;
        }

        pi << setw(4) << right << carry + sum / SCALE;
    }

    auto end = chrono::system_clock::now();
    etime = chrono::duration_cast<chrono::microseconds>(end - start).count();

    printf("Thread %2ld approximated Pi as\t %s\n", tid, pi.str().c_str());
}

int main()
{
    printf("Let's compute pi =\t\t%s\n",
           "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679");

    const int nthreads = 8;
    vector<int> ndigits(nthreads); // number of digits computed by thread tid
    vector<long> etime(nthreads);  // elapsed time of computation

    // Fork a team of threads
#pragma omp parallel num_threads(nthreads)
    // You can use
    // #pragma omp parallel
    // to run with the default number of threads on the node.
    {
        long tid = omp_get_thread_num();

        // Only thread 0 does this
        if (tid == 0)
        {
            int n_threads = omp_get_num_threads();
            printf("[info] Number of threads = %d\n", n_threads);
        }

        // Print the thread ID
        printf("Hello World from thread = %ld\n", tid);

        // Compute digits of pi
        DoWork(tid, ndigits[tid], etime[tid]);
    }
    // All threads join the master thread and terminate

    for (int tid = 0; tid < nthreads; ++tid)
        printf("Thread %d computed %d digits of pi in %5ld musecs (%8.3f musec per digit)\n",
               tid, ndigits[tid], etime[tid],
               static_cast<float>(etime[tid]) / ndigits[tid]);

    return 0;
}
