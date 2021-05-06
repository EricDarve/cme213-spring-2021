---
layout: page
title: Reading Assignment 6
---

## NVIDIA Guest Lecture, openACC and CUDA Optimizations

Write your answers in a PDF and upload the document on [Gradescope](https://www.gradescope.com/courses/258024) for submission. The due date is given on [Gradescope](https://www.gradescope.com/courses/258024). Each question is worth 10 points. 

30 NVIDIA guest lecture, openACC; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=3d931e4b-6a73-481f-b016-ad1d0181ed43); [Slides](../Lecture Slides/CME213_2021_OpenACC.pdf)

Additional resources:

- [OpenACC resources](https://www.openacc.org/resources)
- [OpenACC programming guide](https://www.openacc.org/sites/default/files/inline-files/OpenACC_Programming_Guide_0.pdf)
- OpenACC loop optimizations by Jeff Larkin, Senior DevTech Software Engineer, NVIDIA; [week 1](https://www.openacc.org/sites/default/files/inline-files/OpenACC_Course_Oct2018/OpenACC%20Course%202018%20Week%201.pdf), [week 2](https://www.openacc.org/sites/default/files/inline-files/OpenACC_Course_Oct2018/OpenACC%20Course%202018%20Week%202.pdf), [week 3](https://www.openacc.org/sites/default/files/inline-files/OpenACC_Course_Oct2018/OpenACC%20Course%202018%20Week%203.pdf)
- [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk) (includes an openACC compiler)

Please watch videos 30 and 31 and the slide decks before answering these questions.

1. On slide [39](../Lecture Slides/CME213_2021_OpenACC.pdf#page=39), explain why we have short communication steps on the profiling diagram. Use the code from slide [31](../Lecture Slides/CME213_2021_OpenACC.pdf#page=31). Which variable is being exchanged between the CPU and GPU? Is this data exchanged required? Can it be avoided?
1. Explain the difference between `copy`, `copyin`, and `copyout`.
1. On slide [54](../Lecture Slides/CME213_2021_OpenACC.pdf#page=54), explain what the directive
```
#pragma acc update device(A[0:N])
```
does.
1. On slide [58](../Lecture Slides/CME213_2021_OpenACC.pdf#page=58), explain what the directive
```
#pragma acc kernels loop tile(32, 32)
```
does. Explain how this could have been used in [Homework 4](../Homework/hw4.pdf#page=2). Which implementation variant (Global, Block, Shared) corresponds (approximately) to the openAcc `tile(32,32)` clause?

31 NVIDIA guest lecture, CUDA optimization; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=fa6c8558-b4fd-49cd-9e18-ad1f01843b34); [Slides](../Lecture Slides/CME213_2021_Optimization.pdf)

{:start="5"}
1. Use Little's Law to estimate the achieved bandwidth if on average you have 5 persons on the escalator. Use the parameters on slide [7](../Lecture Slides/CME213_2021_Optimization.pdf#page=7).
1. What are the two conditions that must be satisfied for an instruction to become eligible for issue?
1. For the L2 cache, what is the size of a cache line? What is the size of sector?

Definition: a cache line is a contiguous segment of the cache memory. A directory is used to map a cache line to the corresponding segment in main memory. In a sectored-cache, the cache line is subdivided into sectors. When a cache miss occurs (the data is not found in the cache), only the sector containing the referenced data item is transferred from main memory (rather than the entire cache line). This allows reducing the penalty for cache misses while minimizing the cost of maintaining the cache line directory.
