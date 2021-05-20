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

[Datasheet on the Quadro RTX 6000](Code/quadro.txt)

### Final Project

Final project instructions and starter code:

- [Final Project Part 1](Homework/Part1.pdf)
- [Final Project Part 2](Homework/Part2.pdf)
- [Starter code](Homework/fp.zip)

Slides and videos explaining the final project:

- Overview of the final project; [Slides](Lecture Slides/Lecture_14.pdf)
- 33 Final Project 1, Overview; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=a4426523-7954-4526-a518-ad23017febd1)
- 34 Final Project 2, Regularization; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=c9a4709c-3fe8-434c-a837-ad24000112af)
- 35 Final Project 3, CUDA GEMM and MPI; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=f8578af5-61a1-4b52-89af-ad24001198d4)

See also the [Module 8](https://ericdarve.github.io/cme213-spring-2021/#module-8-group-activity-and-introduction-to-mpi) videos on MPI.

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
- [Reading assignment 3](Reading Assignments/OpenMP_advanced)
- [Homework 2](Homework/hw2.pdf); [starter code](Homework/hw2.zip); [radix sort tutorial](Homework/RadixSortTutorial.pdf)

### Module 4 Introduction to CUDA programming

- Introduction to GPU computing; [Slides](Lecture Slides/Lecture_07.pdf)
- Introduction to CUDA and `nvcc`; [Slides](Lecture Slides/Lecture_08.pdf); [Code](https://github.com/EricDarve/cme213-spring-2021/tree/main/Code/Lecture_08)
- 19 GPU computing introduction; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=6d471ac2-3cf2-4adb-b8b9-ad0a016ce37d)
- 20 Graphics Processing Units; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=241f3349-9454-4cca-833a-ad0a017729a1)
- 21 Introduction to GPU programming; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=e3277879-f1b8-4f14-8d87-ad0a01811c60)
- 22 icme-gpu; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=0cddfda0-dd3e-4315-b669-ad0c016b3ce6)
- 23 a First CUDA program; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=1c2baeca-880d-49b2-a993-ad0c018b2ed9)
- 23 b First CUDA program part 2; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=6d7e66bf-5b82-4200-bfd6-ad0d0006288a)
- 24 nvcc CUDA compiler; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=e6436528-7096-4b7e-804d-ad0d011ff2f9)
- [Reading assignment 4](Reading Assignments/CUDA_intro)
- [Homework 3](Homework/hw3.pdf); [starter code](Homework/hw3.zip)

### Module 5 Code performance on NVIDIA GPUs

- GPU memory and matrix transpose; [Slides](Lecture Slides/Lecture_09.pdf); [Code](https://github.com/EricDarve/cme213-spring-2021/tree/main/Code/Lecture_09)
- CUDA occupancy, branching, homework 4; [Slides](Lecture Slides/Lecture_10.pdf)
- 25 GPU memory; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=a337c354-634b-4036-a2bd-ad120000d2dc)
- 26 Matrix transpose; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=a2ae4f59-bdbb-4a67-a468-ad120023855e)
- 27 Latency, concurrency, and occupancy; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=78a55dba-48a6-474b-bb69-ad13000fd03f)
- 28 CUDA branching; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=bf1019a1-0440-4d70-951b-ad130028ef48)
- 29 Homework 4; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=8e103553-aedf-434c-a1c5-ad1400016f4c)
- [Reading assignment 5](Reading Assignments/GPU_performance)
- [Homework 4](Homework/hw4.pdf); [starter code](Homework/hw4.zip)

### Module 6 NVIDIA guest lectures, openACC, CUDA optimization

- 30 NVIDIA guest lecture, openACC; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=3d931e4b-6a73-481f-b016-ad1d0181ed43); [Slides](Lecture Slides/CME213_2021_OpenACC.pdf)
- 31 NVIDIA guest lecture, CUDA optimization; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=fa6c8558-b4fd-49cd-9e18-ad1f01843b34); [Slides](Lecture Slides/CME213_2021_Optimization.pdf)
- [Reading assignment 6](Reading Assignments/NVIDIA_openACC_optimization)

### Module 7 NVIDIA guest lectures, CUDA profiling

- 32 NVIDIA guest lecture, CUDA profiling; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=ba6a97a2-7841-4c37-8ca1-ad240180023e); [Slides](Lecture Slides/CME213_2021_CUDA_Profiling.pdf)
- [Reading assignment 7](Reading Assignments/NVIDIA_CUDA_profiling)

### Module 8 Group activity and introduction to MPI

The slides and videos below are needed for the final project.

- Introduction to MPI; [Slides](Lecture Slides/Lecture_16.pdf); [Code](https://github.com/EricDarve/cme213-spring-2021/tree/main/Code/Lecture_16)
- 37 MPI Introduction; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=c7b9b334-fbac-4c51-9740-ad29017847d2)
- 38 MPI Hello World; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=c8beedcb-bbc4-439b-b1d1-ad29017e0914)
- 39 MPI Send Recv; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=d36f28c6-9c62-4c80-8902-ad29018641d6)
- 40 MPI Collective Communications; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=51c5cb27-157b-4fe4-a264-ad2a00028c67)

Material for the May 17 group activity:

- [generate_sequence.cpp](Code/generate_sequence.cpp)
- 36 Instructions for Monday, May 17 group activity; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=04de4abf-6adb-4c3a-ade1-ad27002d0260); [Slides](Lecture Slides/Lecture_15.pdf)

### Module 9 Advanced MPI

- MPI Advanced Send and Recv; [Slides](Lecture Slides/Lecture_17.pdf); [Code](https://github.com/EricDarve/cme213-spring-2021/tree/main/Code/Lecture_17)
- 41 MPI Process Mapping; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=53a88ac4-8574-4c83-8c2b-ad2d00031dff)
- 42 MPI Buffering; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=36217368-59a9-416a-b854-ad2d0007de35)
- 43 MPI Send Recv Deadlocks; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=d662dbbc-18fb-4efb-bb02-ad2d000b10c1)
- 44 MPI Non-blocking; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=1a68314e-15ab-4ac9-a8a0-ad2d000ebc31)
- 45 MPI Send Modes; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=23b20f84-badf-4463-8a93-ad2d0029dd5f)
- 46 MPI Matrix-vector product 1D schemes; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=40c04c7e-ee19-45d0-8bb0-ad2d01784ff6)
- 47 MPI Matrix vector product 2D scheme; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=d4e398db-28eb-4747-a032-ad2d017dfc16)
- 48 Parallel Speed-up; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=dd191579-3c13-4e79-9d2a-ad2d01824d85)

<!-- - [Reading assignment 8](Reading Assignments/MPI) -->

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
