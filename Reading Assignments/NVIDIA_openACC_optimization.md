---
layout: page
title: Reading Assignment 6
---

## NVIDIA Guest Lecture, openACC and CUDA Optimizations

Write your answers in a PDF and upload the document on [Gradescope](https://www.gradescope.com/courses/258024) for submission. The due date is given on [Gradescope](https://www.gradescope.com/courses/258024). Each question is worth 10 points. 

Please watch the videos 30 through 31 and the slides before answering these questions.

[Slide Deck](../Lecture Slides/CME213_2021_OpenACC.pdf)

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