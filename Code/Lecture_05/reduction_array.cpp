#include <omp.h>
#include <vector>
#include <cassert>
#include <iostream>
using namespace std;

int main(void)
{
    const int n = 512;

    float b[n][n];
    float a[n];

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        a[i] = 0;
        for (int j = 0; j < n; j++)
            b[i][j] = i + (j % 5);
    }

    // Example 1: reduction with an array
    #pragma omp parallel for reduction(+: a [0:n])
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            a[j] += b[i][j];

    // Test
    for (int j = 0; j < n; j++)
    {
        float sum = 0;
        for (int i = 0; i < n; i++)
            sum += b[i][j];
        assert(sum == a[j]);
    }

    // Example 2: this loop does not require a reduction clause

    // Re-initialize
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        a[i] = 0;

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            a[i] += b[i][j];

    // Test
    for (int i = 0; i < n; i++)
    {
        float sum = 0;
        for (int j = 0; j < n; j++)
            sum += b[i][j];
        assert(sum == a[i]);
    }

    cout << "PASS\n";
    return 0;
}
