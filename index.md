---
layout: page
title: Stanford CME 213/ME 339 Spring 2021 homepage
---

**Introduction to parallel computing using MPI, openMP, and CUDA**

This is the website for CME 213 *Introduction to parallel computing using MPI, openMP, and CUDA.* This material was created by [Eric Darve](https://me.stanford.edu/people/eric-darve), with the help of course staff and students.

## Syllabus

[Syllabus](syllabus)

## Policy for late assignments

Extensions can be requested in advance for exceptional circumstances (e.g., travel, sickness, injury, COVID-related issues) and for OAE-approved accommodations.

Submissions after the deadline and late by at most two days (+48 hours after the deadline) will be accepted with a 10% penalty. No submissions will be accepted two days after the deadline.

See [Gradescope](https://www.gradescope.com/courses/258024) for all the current assignments and their due dates. Post on [Slack](https://cme213-spring-2021.slack.com/) if you cannot access the Gradescope class page. The 6-letter code to join the class is given on [Canvas](https://canvas.stanford.edu/courses/133903).

## Class modules and learning material

### Introduction to the class

CME 213 First Live Lecture; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=b2bea53e-710b-443e-bb1f-acfa0178a41b), [Slides](Lecture Slides/Lecture_01.pdf)

### C++ tutorial

- [Tutorial slides](Lecture Slides/cpp tutorial/Tutorial_01.pdf)
- [Tutorial code](Lecture Slides/cpp tutorial/code.zip)

### Module 1 Introduction to Parallel Computing

- [Slides](Lecture Slides/Lecture_02.pdf)
- 01 Homework 1; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=0be0c85d-3bba-4758-8477-acfa011936d3)
- 02 Why Parallel Computing; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=242a8ba4-239c-4233-ab47-acfa011bc3c8)
- 03 Top 500; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=344eb483-fdb4-4c2a-b19b-acfa012393bd)
- 04 Example of Parallel Computation; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=27f038d0-7ca5-4cea-baf7-acfa012640b5)
- 05 Shared memory processor; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=95db1601-8ae0-45af-93a1-acfa012c4919)
- [Reading assignment 1](Reading Assignments/Introduction_Parallel_Computing)
- [Homework 1](Homework/hw1.pdf); [starter code](Homework/hw1.zip)

### Module 2 Shared Memory Parallel Programming

- C++ threads; [Slides](Lecture Slides/Lecture_03.pdf); [Code](https://github.com/EricDarve/cme213-spring-2021/tree/main/Code/Lecture_03)
- Introduction to OpenMP; [Slides](Lecture Slides/Lecture_04.pdf); [Code](https://github.com/EricDarve/cme213-spring-2021/tree/main/Code/Lecture_04)
- 06 C++ threads; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=f9bc07e2-9ed3-4659-b884-acff017ff497)
- 07 Promise and future; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=79f84744-e1c9-47dc-a832-acff018a68f9)
- 08 mutex; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=a6a16c1e-bcb7-4e7c-b70d-ad000005d4b3)
- 09 Introduction to OpenMP; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=29287cfe-d491-4d39-a861-ad000121a591)
- 10 OpenMP Hello World; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=4d946d55-6cc9-4b20-a7ad-ad0001249d0b)
- 11 OpenMP for loop; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=1ec28d98-e910-4458-bac8-ad00012ce56a)
- 12 OpenMP clause; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=daee4a50-36fa-48f1-a244-ad000138d1c7)
- [Reading assignment 2](Reading Assignments/OpenMP)

### Module 3 Shared Memory Parallel Programming, OpenMP, advanced OpenMP

- OpenMP, for loops, advanced OpenMP; [Slides](Lecture Slides/Lecture_05.pdf); [Code](https://github.com/EricDarve/cme213-spring-2021/tree/main/Code/Lecture_05)
- OpenMP, sorting algorithms; [Slides](Lecture Slides/Lecture_06.pdf); [Code](https://github.com/EricDarve/cme213-spring-2021/tree/main/Code/Lecture_06)
- 13 OpenMP tasks; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=35aeb01b-2bde-4874-8e5c-ad040000c4d9)
- 14 OpenMP depend; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=177e97ab-0811-48c6-8f43-ad04000dc0e3)
- 15 OpenMP synchronization; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=578d24ec-4323-4e21-a8d4-ad0400267df5)
- 16 Sorting algorithms Quicksort Mergesort; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=368a4b90-eb0f-4e78-bf4e-ad05017f6de7)
- 17 Sorting Algorithms Bitonic Sort; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=2c1c540a-d757-4dd1-b9ae-ad060004de65)
- 18 Bitonic Sort Exercise; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=29690a03-da5e-468c-bdfb-ad060014f654)
- [Homework 2](Homework/hw2.pdf); [starter code](Homework/hw2.zip); [radix sort tutorial](Homework/RadixSortTutorial.pdf)

## Reading and links

**Lawrence Livermore National Lab Resources**

- [LLNL Tutorial and Training Materials](https://hpc.llnl.gov/training/tutorials)
- [LLNL Introduction to Parallel Computing tutorial](https://hpc.llnl.gov/training/tutorials/introduction-parallel-computing-tutorial)
- [LLNL POSIX threads programming](https://hpc-tutorials.llnl.gov/posix/)
- [LLNL openMP tutorial](https://hpc.llnl.gov/openmp-tutorial)
- [LLNL MPI tutorial](https://hpc-tutorials.llnl.gov/mpi/)
- [LLNL Advanced MPI slides](https://hpc.llnl.gov/sites/default/files/DavidCronkSlides.pdf)

#### C++ threads

- [C++ reference](https://en.cppreference.com/w/cpp)
- [Simple examples of C++ multithreading](https://www.geeksforgeeks.org/multithreading-in-cpp/)
- [C++ threads](https://en.cppreference.com/w/cpp/thread/thread/thread)
- [LLNL tutorial on Pthreads](https://computing.llnl.gov/tutorials/pthreads/)

#### OpenMP

- [OpenMP LLNL guide](https://computing.llnl.gov/tutorials/openMP/)
- [OpenMP guide by Yliluoma](https://bisqwit.iki.fi/story/howto/openmp/)
- [OpenMP 5.0 Reference Guide](https://www.openmp.org/wp-content/uploads/OpenMPRef-5.0-0519-web.pdf)
- [OpenMP API Specification](https://www.openmp.org/spec-html/5.1/openmp.html)
- [Tutorials](https://www.openmp.org/resources/tutorials-articles/)

#### CUDA

- [CUDA Programming Guides and References](http://docs.nvidia.com/cuda/index.html)
- [CUDA C++ Programming Guide](http://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf)
- [CUDA C++ Best Practices Guide](http://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf)
- [CUDA occupancy calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/CUDA_Occupancy_Calculator.xls)
- [CUDA compiler driver NVCC](https://docs.nvidia.com/cuda/pdf/CUDA_Compiler_Driver_NVCC.pdf)
- [OpenACC](https://www.openacc.org/)
- [OpenACC Programming and Best Practices Guide](https://www.openacc.org/sites/default/files/inline-files/OpenACC_Programming_Guide_0.pdf)
- [OpenACC 2.7 API Reference Card](https://www.pgroup.com/lit/literature/openacc-api-guide-2.7.pdf)
- [Compilers that support OpenACC](https://www.openacc.org/tools)
- [OpenACC Specification (Version 3.0)](https://www.openacc.org/sites/default/files/inline-images/Specification/OpenACC.3.0.pdf)

#### MPI

- [MPI standard version 3.1](https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report.pdf)
- [Open MPI documentation](https://www-lb.open-mpi.org/doc/current/)
- [Mapping, Ranking, and Binding](https://www-lb.open-mpi.org/doc/current/man1/mpirun.1.php#sect12)

**Open MPI hwloc documentation**

- [Hwloc tutorial slides](https://www-lb.open-mpi.org/projects/hwloc/tutorials/20160606-PATC-hwloc-tutorial.pdf)
- [Open-mpi hwloc documentation page](https://www-lb.open-mpi.org/projects/hwloc/)
- [Hwloc/lstopo examples](https://www.open-mpi.org/projects/hwloc/doc/v2.1.0/a00328.php#cli_examples)

#### Task-based parallel languages and APIs

- [Legion](https://legion.stanford.edu/) and [Regent](https://regent-lang.org/)
- [StarPU](https://starpu.gitlabpages.inria.fr/)
- [Charm++](https://charmplusplus.org/)
- [PaRSEC](https://icl.utk.edu/parsec/index.html)
- [Chapel](https://chapel-lang.org/)
- [X10](http://x10-lang.org/)
- [TaskTorrent](https://github.com/leopoldcambier/tasktorrent) and [documentation](https://tasktorrent.readthedocs.io/en/latest/)

#### Sorting algorithms

- [A novel sorting algorithm for many-core architectures based on adaptive bitonic sort](https://ieeexplore.ieee.org/abstract/document/6267838)
- [Adaptive Bitonic Sorting](https://pdfs.semanticscholar.org/bcdf/c4e40c79547c9daf89dada4e1c23056871cb.pdf)
