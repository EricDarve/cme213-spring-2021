---
layout: page
title: Reading Assignment 6
---

## NVIDIA Guest Lecture, openACC and CUDA Optimizations

Write your answers in a PDF and upload the document on [Gradescope](https://www.gradescope.com/courses/258024) for submission. The due date is given on [Gradescope](https://www.gradescope.com/courses/258024). Each question is worth 10 points. 

[Slide Deck](../Lecture Slides/CME213_2021_OpenACC.pdf)

Additional resources:

- [OpenACC resources](https://www.openacc.org/resources)
- [OpenACC programming guide](https://www.openacc.org/sites/default/files/inline-files/OpenACC_Programming_Guide_0.pdf)
- OpenACC loop optimizations by Jeff Larkin, Senior DevTech Software Engineer, NVIDIA; [week 1](https://www.openacc.org/sites/default/files/inline-files/OpenACC_Course_Oct2018/OpenACC%20Course%202018%20Week%201.pdf), [week 2](https://www.openacc.org/sites/default/files/inline-files/OpenACC_Course_Oct2018/OpenACC%20Course%202018%20Week%202.pdf), [week 3](https://www.openacc.org/sites/default/files/inline-files/OpenACC_Course_Oct2018/OpenACC%20Course%202018%20Week%203.pdf)
- [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk) (includes an openACC compiler)

Please watch videos 30 and 31 and the slide deck before answering these questions.

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


This reading assignment will be continued after the Wednesday, May 5 lecture has concluded.