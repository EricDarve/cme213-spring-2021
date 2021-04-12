---
layout: page
title: Reading Assignment 2
---

## OpenMP

Write your answers in a PDF and upload the document on [Gradescope](https://www.gradescope.com/courses/258024) for submission. The due date is given on [Gradescope](https://www.gradescope.com/courses/258024). Each question is worth 10 points. 

[Starter code](../Code/ra2.zip) for all the questions.

Please watch the videos 06 through 12 and the slides before answering the questions:

- C++ threads; [Slides](../Lecture Slides/Lecture_03.pdf); [Code](https://github.com/EricDarve/cme213-spring-2021/tree/main/Code/Lecture_03)
- Introduction to OpenMP; [Slides](../Lecture Slides/Lecture_04.pdf); [Code](https://github.com/EricDarve/cme213-spring-2021/tree/main/Code/Lecture_04)
- 06 C++ threads; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=f9bc07e2-9ed3-4659-b884-acff017ff497)
- 07 Promise and future; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=79f84744-e1c9-47dc-a832-acff018a68f9)
- 08 mutex; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=a6a16c1e-bcb7-4e7c-b70d-ad000005d4b3)
- 09 Introduction to OpenMP; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=29287cfe-d491-4d39-a861-ad000121a591)
- 10 OpenMP Hello World; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=4d946d55-6cc9-4b20-a7ad-ad0001249d0b)
- 11 OpenMP for loop; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=1ec28d98-e910-4458-bac8-ad00012ce56a)
- 12 OpenMP clause; [Video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=daee4a50-36fa-48f1-a244-ad000138d1c7)

Answer these questions:

1. In [`cpp_thread.cpp`](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_03/cpp_thread.cpp), complete the exercise with function `f4`. Your code should pass the `assert`. Turn in your code.
1. In [`cpp_thread.cpp`](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_03/cpp_thread.cpp), complete the exercise with `t6` and `max_result`. Your code should pass the `assert`. Turn in your code.
1. In [`mutex_demo.cpp`](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_03/mutex_demo.cpp), explain what would happen if you remove line [45](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_03/mutex_demo.cpp#L45): `g_mutex.unlock();`
1. In [`hello_world_openmp.cpp`](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_04/hello_world_openmp.cpp), explain what the line [60](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_04/hello_world_openmp.cpp#L60) does: `#pragma omp parallel num_threads(nthreads)`. Explain the `num_threads` clause.
1. Using an OpenMP pragma, modify the file [`matrix_prod_openmp.cpp`](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_04/matrix_prod_openmp.cpp) such that the execution of the code between line [90](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_04/matrix_prod_openmp.cpp#L90) and [99](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_04/matrix_prod_openmp.cpp#L99) is accelerated by using multiple threads. Report on your running times as you vary the number of threads. Use the options `-p` and `-n`. Turn in your code.
1. In [shared_private_openmp.cpp](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_04/shared_private_openmp.cpp), explain what would happen if you remove `private(is_private)` on line [31](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_04/shared_private_openmp.cpp#L31).