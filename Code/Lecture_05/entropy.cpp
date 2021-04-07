#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <omp.h>
using namespace std;

void print_vector(vector<float> &vec)
{
    const unsigned n = vec.size();
    printf("(");
    for (unsigned i = 0; i < n - 1; ++i)
        printf("%7.4f,", vec[i]);
    printf("%7.4f)\n", vec[n - 1]);
}

int main(void)
{
    const unsigned size = 16;
    vector<float> entropy0(size), entropy1(size);

    // todo: parallel for
    for (unsigned i = 0; i < size; i++)
    {
        float d = i - float(size - 1) / 2.;
        entropy0[i] = exp(-d * d / 0.5);
        entropy1[i] = exp(-d * d / 10.);
    }

    float sum1 = 0, sum2 = 0;
    // todo: parallel for + reduction on sum1 & sum2
    // syntax: reduction(+: var1, var2, ...)
    for (unsigned i = 0; i < size; i++)
    {
        sum1 += entropy0[i];
        sum2 += entropy1[i];
    }

    // todo: parallel for
    for (unsigned i = 0; i < size; i++)
    {
        entropy0[i] /= sum1;
        entropy1[i] /= sum2;
    }

    print_vector(entropy0);
    print_vector(entropy1);

    float ent1 = 0, ent2 = 0;
    // todo: parallel for + reduction on ent1 & ent2
    for (unsigned i = 0; i < size; i++)
    {
        if (entropy0[i] > 0)
            ent1 += entropy0[i] * log(entropy0[i]);
        if (entropy1[i] > 0)
            ent2 += entropy1[i] * log(entropy1[i]);
    }

    ent1 = -ent1;
    ent2 = -ent2;

    printf("Change in entropy: %g\n", ent2 - ent1);
    assert(fabs(ent2 - ent1 - 1.4378435611724853515625) < 1e-6);

    return 0;
}
