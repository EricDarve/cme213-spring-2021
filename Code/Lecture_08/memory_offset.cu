#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <unistd.h>
#include <utils.h>
#include <vector>

using std::vector;

__global__
void initialize(int n, float* a) {
    int xid = blockIdx.x * blockDim.x + threadIdx.x;
    if (xid < n)
        a[xid] = xid % 1024;
}

__global__ void offsetCopy(int n, float* odata, float* idata, int offset) {
    int wid = threadIdx.x/32;
    int xid = blockIdx.x * 2 * blockDim.x + 32 * wid + threadIdx.x + offset;
    if (xid < n)
        odata[xid] = idata[xid];
}

__global__ void stridedCopy(int n, float* odata, float* idata, int stride) {
    int xid = stride * (blockIdx.x * blockDim.x + threadIdx.x);
    if (xid < n)    
        odata[xid] = idata[xid];
}

__global__ void randomCopy(int n, float* odata, float* idata, int* addr) {
    int xid = blockIdx.x * blockDim.x + threadIdx.x;
    if (xid < n && addr[xid] < n)    
        odata[xid] = idata[addr[xid]];
}

__global__ void tid(int * warpID) {
    int tID = threadIdx.x 
        + threadIdx.y * blockDim.x 
        + threadIdx.z * blockDim.x * blockDim.y;

    *warpID = tID/32;
}

int main(void) {

    const int one_G = 1<<30;
    const int n = one_G / 8;
    const int n_thread = 512;
    float* d_a, *d_b;

    /* Allocate memory */
    checkCudaErrors(cudaMalloc(&d_a, sizeof(float) * n));
    checkCudaErrors(cudaMalloc(&d_b, sizeof(float) * n));

    printf("Number of bytes copied %ldu\n", sizeof(float) * n);

    int n_blocks = (n + n_thread - 1) / n_thread;

    printf("Matrix size: %d; number of threads per block: %d; number of blocks: %d\n", 
        n, n_thread, n_blocks);    

    initialize<<<n_blocks, n_thread>>>(n, d_a);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    GpuTimer timer;

    // Benchmarks

    vector<int> offset ={0,1,2,32,64,128};

    for (auto o : offset) {
        timer.start();  
        for (int num_runs = 0; num_runs < 100; ++num_runs) {
            offsetCopy<<<n_blocks, n_thread>>>(n, d_b, d_a, o);
        }
        timer.stop();
        printf("Elapsed time for offset %3d in msec: %10.4f\n",o,timer.elapsed() / 100);             
    }

    vector<int> stride ={1,2,4,8,16};

    for (auto s : stride) {
        timer.start();  
        for (int num_runs = 0; num_runs < 100; ++num_runs) {
            stridedCopy<<<n_blocks, n_thread>>>( (n*s)/16, d_b, d_a, s);
        }
        timer.stop();
        printf("Elapsed time for stride %3d in msec: %10.4f\n",s,timer.elapsed() / 100);             
    }    

    return 0;
}
