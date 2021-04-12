---
layout: page
title: Reading Assignment 3
---

## OpenMP Advanced

Write your answers in a PDF and upload the document on [Gradescope](https://www.gradescope.com/courses/258024) for submission. The due date is given on [Gradescope](https://www.gradescope.com/courses/258024). Each question is worth 10 points. 

Please watch the videos 13 through 18 and the slides before answering these questions.

[Starter code](../Code/ra3.zip) for all the questions.

[Slide Deck](../Lecture Slides/Lecture_05.pdf)

1. Why is
```
#pragma omp task
```
typically preceded by
```
#pragma omp single
```
?
1. In OpenMP, `firstprivate` variables must be initialized inside the task. True/False? Explain.
1. In [`tree_postorder.cpp`](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_05/tree_postorder.cpp), explain why we use the `shared` clause on line [65](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_05/tree_postorder.cpp#L65) and [69](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_05/tree_postorder.cpp#L69).

For the next questions, please open slide [29](../Lecture Slides/Lecture_05.pdf#page=29) in `Lecture_05.pdf`. Then answer these questions:

{:start="4"}
1. Task T2 comes after T1. Always/Never/Sometimes. Explain.
1. Task T2 comes after T4. Always/Never/Sometimes. Explain.
1. Task T5 comes after T1. Always/Never/Sometimes. Explain.
1. Tasks T4 and T5 may run concurrently. Yes/No/It depends. Explain.
1. Task T5 comes after T1 and T3. Always/Never/Sometimes. Explain.
1. This program prints out "d=6". Yes/No/It depends. Explain.
1. When both are correct, atomic is preferred over reduction in a for loop. True/False? Explain.
1. Complete the todo in [`entropy.cpp`](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_05/entropy.cpp). See lines [22](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_05/entropy.cpp#L22), [31](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_05/entropy.cpp#L31), [39](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_05/entropy.cpp#L39), and [50](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_05/entropy.cpp#L50). Turn in your code.

[Slide Deck](../Lecture Slides/Lecture_06.pdf). Make sure you watch the corresponding [video](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=29690a03-da5e-468c-bdfb-ad060014f654).

{:start="12"}
1. Parallelize [`bitonic_sort_lab.cpp`](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_06/bitonic_sort_lab.cpp) using OpenMP. Follow the instructions in the slides. You should make at least the following changes: parallelize the `j` loop on line [86](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_06/bitonic_sort_lab.cpp#L86). Split the `i` loop on line [84](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_06/bitonic_sort_lab.cpp#L84). Modify the function `BitonicSortPar` to parallelize the `i` loop on line [169](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_06/bitonic_sort_lab.cpp#L169). Modify line [184](https://github.com/EricDarve/cme213-spring-2021/blob/main/Code/Lecture_06/bitonic_sort_lab.cpp#L184) by adding an `if` statement. Add and parallelize the `j` loop as explained on slide [44](../Lecture Slides/Lecture_06.pdf#page=44). At the end, you should have three new `#pragma omp` directives. Turn in your code. Turn in the output of your code **without** the `-DNDEBUG` compile option.

If you compile your code without `-DNDEBUG`, this will keep the `assert` at the end. If you implement your code correctly, you should pass the `assert` test at the end.

With the `-DNDEBUG` option, the code will sort a longer array. You should see a speed up when using multiple threads. Run your code using

```
export OMP_NUM_THREADS=4; ./bitonic_sort
```