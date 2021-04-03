#include <omp.h>

#include <cassert>
#include <vector>
#include <iostream>

using namespace std;

template <typename T>
void print_vector(T &vec)
{
    const unsigned n = vec.size();
    cout << "(";
    for (unsigned i = 0; i < n - 1; ++i)
        cout << vec[i] << ", ";
    cout << vec[n - 1] << ")\n";
}

int main(int argc, char **argv)
{
    const int n = 8;

    vector<float> x(n), y(n), z(n);

#pragma omp parallel for
    for (int i = 0; i < n; ++i)
    {
        x[i] = i;
        y[i] = 2 * i;
    }

    print_vector(x);
    print_vector(y);

    for (int i = 0; i < n; ++i)
        assert(x[i] == i && y[i] == 2 * i);

#pragma omp parallel for
    for (int i = 0; i < n; ++i)
        z[i] = x[i] + y[i];

    print_vector(z);

    for (int i = 0; i < n; ++i)
        assert(z[i] == x[i] + y[i]);

    return 0;
}