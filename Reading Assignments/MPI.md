---
layout: page
title: Reading Assignment 8
---

## MPI

Write your answers in a PDF and upload the document on [Gradescope](https://www.gradescope.com/courses/258024) for submission. The due date is given on [Gradescope](https://www.gradescope.com/courses/258024). Each question is worth 10 points. 

Please watch the videos and slides starting from video "37 MPI Introduction" to video "50 MPI Communicators" (Module 8 and 9).

1. Define the acronyms SIMD and SPMD.
2. Explain what `MPI_ANY_SOURCE` does.
3. What are tags used for in `MPI_Send` and `MPI_Recv`?
4. Explain what the MPI functions `MPI_Allgather` and `MPI_Alltoall` do.
5. Explain what the following command does:

```
mpirun --bind-to hwthread --map-by core ./mpi_hello
```

{:start="6"}
6. Explain the difference between `MPI_Recv` with and without buffer.
7. Describe a scenario in which an MPI program deadlocks.
8. Explain the role of `MPI_Request` when doing `MPI_Isend` and `MPI_Irecv`.
9. Is there a performance difference between an MPI matrix-vector product with row partitioning and column partitioning?
10. What is the speed-up of a program according to Amdahl's Law?
11. What is the speed-up of a program according to Gustafson's Law?
12. How does the efficiency typically change as you increase the number of processes?
13. What is the running time of an optimal all-to-all personalized collective operation with a hypercube network?
14. In a matrix-matrix product, the total number of operations is $O(n^3)$. Assuming we have $p$ MPI processes, the total number of flops per process is $O(n^3/p)$. If we use the Dekel-Nassimi-Sahni algorithm, accounting for communications, we have the following running times:

$$ T_1(n) = \alpha n^3 $$

$$ T_p(n) = \alpha \frac{n^3}{p} + \beta \log p + \gamma \Big( \frac{n^2}{p^{2/3}} \Big) \log p $$

Compute the iso-efficiency function $p(n)$ of this algorithm.

Hint: slide [60](https://ericdarve.github.io/cme213-spring-2021/Lecture%20Slides/Lecture_18.pdf#page=60) will be useful.

{:start="15"}
15. Explain what the MPI function `MPI_Group_incl` does.

