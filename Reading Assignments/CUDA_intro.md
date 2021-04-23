---
layout: page
title: Reading Assignment 4
---

## Introduction to CUDA Programming

Write your answers in a PDF and upload the document on [Gradescope](https://www.gradescope.com/courses/258024) for submission. The due date is given on [Gradescope](https://www.gradescope.com/courses/258024). Each question is worth 10 points. 

Please watch the videos 19 through 24 and the slides before answering these questions.

<!-- [Starter code](../Code/ra3.zip) for all the questions. -->

[Slide Deck](../Lecture Slides/Lecture_07.pdf)

1. Describe three features that differentiate CPU from GPU processors.

<!-- 1. CPU cores are more complex than GPU cores
1. GPU have a lot more cores
1. CPU cache are much larger
1. CPUs typically have an L3 cache
1. The control on a CPU is more complex than on a GPU -->

{:start="2"}
1. What is the double precision performance of a Quadro RTX 6000 compared to its single precision performance?

1. Assume you launch a CUDA kernel from the CPU code. When the function call returns on the CPU, does it mean that the CUDA kernel execution has completed on the GPU?

1. What is an NVIDIA tensor core?

1. How many SMs are required to run a CUDA thread block? Does the answer depend on the number of threads in the block?

[Slide Deck](../Lecture Slides/Lecture_08.pdf)

{:start="6"}
1. Explain the difference between `sbatch` and `srun` in SLURM.
1. What is the SLURM command to cancel a job?
1. Explain the meaning of the keywords `__global__` and `__device__` in CUDA.
1. Explain what the following built-in CUDA variables are: `threadIdx`, `blockDim`, `blockIdx`.
1. [Starter code.](../Code/ra4.zip) Read the program [`firstProgram.cu`](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_08/firstProgram.cu). Then, fill-in the TODOs in `R4.cu` (contained in the zip file) so that you compute an array of type `float` with entries
```
out[i] = 1. / i;
```
Please read as well [`addMatrices.cu`](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_08/addMatrices.cu) where you will find useful examples. The size of the array should be equal to 100,000. Each CUDA thread should compute a single entry `out[i]`. The number of threads in a CUDA block should be chosen equal to 512.
1. Explain the difference between a virtual architecture and a real architecture in `nvcc`.
1. What are the recommended `nvcc` options to compile CUDA code on `icme-gpu`?
1. Explain what the shorthand option `--gpu-architecture=sm_75` does during the compilation process using `nvcc`.