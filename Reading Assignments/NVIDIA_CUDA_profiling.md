---
layout: page
title: Reading Assignment 7
---

## NVIDIA Guest Lecture, CUDA profiling

Write your answers in a PDF and upload the document on [Gradescope](https://www.gradescope.com/courses/258024) for submission. The due date is given on [Gradescope](https://www.gradescope.com/courses/258024). Each question is worth 10 points. 

32 NVIDIA guest lecture, CUDA profiling; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=ba6a97a2-7841-4c37-8ca1-ad240180023e); [Slides](../Lecture Slides/CME213_2021_CUDA_Profiling.pdf)

[NVIDIA Developer Blog on the high-performance multigrid CUDA code (HPGMG)](https://developer.nvidia.com/blog/high-performance-geometric-multi-grid-gpu-acceleration/)

1. Explain the difference between kernels that are compute bound, bandwidth bound, and latency bound
1. See the right figure on Slide [25](../Lecture Slides/CME213_2021_CUDA_Profiling.pdf#page=25); what is a long scoreboard stall? See the NVIDIA document, [section 4.1](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#statistical-sampler) for some definitions; [scoreboarding](https://en.wikipedia.org/wiki/Scoreboarding) is a method to track dependencies between instructions and is used to determine when a warp is ready to run.
1. Assume that we have a pipeline with depth 10 cycles; assume that we have 5 warps that can issue instructions in parallel; on average, how many instructions is the pipeline able to issue per cycle? See slide [27](../Lecture Slides/CME213_2021_CUDA_Profiling.pdf#page=27).
1. Consider the following codes:

Version 1:
```
float a = 0.0f;
for( int i = 0 ; i < N ; ++i )
 a += logf(b[i]);
```

Version 2:
```
float a, a0 = 0.0f, a1 = 0.0f;
for( int i = 0 ; i < N ; i += 2 )
{
 a0 += logf(b[i]);
 a1 += logf(b[i+1]);
}
a += logf(c) a = a0 + a1
```

Explain why version 2 is expected to run faster on a GPU.

{:start="5"}
1. Explain the difference between [Nsight Systems](https://developer.nvidia.com/nsight-systems) and [Nsight Compute](https://developer.nvidia.com/nsight-compute).