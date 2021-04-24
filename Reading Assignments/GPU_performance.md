---
layout: page
title: Reading Assignment 5
---

## GPU Performance

Write your answers in a PDF and upload the document on [Gradescope](https://www.gradescope.com/courses/258024) for submission. The due date is given on [Gradescope](https://www.gradescope.com/courses/258024). Each question is worth 10 points. 

Please watch the videos 25 through 28 and the slides before answering these questions.

[Slide Deck](../Lecture Slides/Lecture_09.pdf)

1. Is the L2 cache local to an SM or is it shared by all SMs on the GPU?
1. See slide [11](../Lecture Slides/Lecture_09.pdf#page=11). On the Turing GPU, which offsets give you peak bandwidth and which offsets give you the lowest bandwidth?
1. See slide [27](../Lecture Slides/Lecture_09.pdf#page=27). Consider `simpleTranspose`. Assume that the read is at peak performance (480 GB/sec). Assume that the writes have the worst possible performance (1/32 of the peak). Estimate the performance of `simpleTranspose`. Does your estimate match the measurement?
1. Let's assume that thread `i` in a warp accesses bank `(3*i)%32` in shared memory. Does this lead to a shared memory bank conflict? What happens if thread `i` accesses bank `(6*i)%32`?


[Slide Deck](../Lecture Slides/Lecture_10.pdf)

{:start="5"}
1. Using the [CUDA occupancy calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/CUDA_Occupancy_Calculator.xls), calculate the occupancy (in %) for a Turing GPU if 80 registers per thread are required and you have 512 threads per block. Assume that the kernel is not using any shared memory. What is the maximum register count with this configuration (512 threads per block) that would give you 100% occupancy?
1. Consider the following code:
```
__global__ void branch(float* out){
    if (threadIdx.x >= 256) {
        ...;
    } else {
        ...;
    }
}
```
Does this code lead to warp divergence?