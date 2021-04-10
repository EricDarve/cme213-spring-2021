import numpy as np

def pretty_print(A,clr):
    print('[',end=' ')
    for i in range(A.size):
        if clr[i] == 1:
            print('\033[91m',A[i],end=' ')
        elif clr[i] == 2:
            print('\033[94m',A[i],end=' ')
        elif clr[i] == 3:
            print('\033[92m',A[i],end=' ')
        elif clr[i] == -1:
            print('\033[37m',A[i],end=' ')
        else:
            print('\033[0m',A[i],end=' ')
    print('\033[0m',']')

def quicksort(A,l,u):
    if l < u-1:
        x = A[l]
        s = l
        clr = np.zeros(A.size)
        clr[:] = -1
        clr[l:u] = 0
        clr[l] = 2
        for i in range(l+1,u):
            if A[i] <= x: # Swap entries smaller than pivot
                s = s+1
                A[s], A[i] = A[i], A[s]
                # Next 3 lines for printing
                clr[s] = clr[i] = 1
                pretty_print(A,clr)
                clr[s] = clr[i] = 0
        A[s], A[l] = A[l], A[s]
        clr[l] = 1; clr[s] = 2
        pretty_print(A,clr)
        clr[l] = clr[s] = 0
        quicksort(A,l,s)
        quicksort(A,s+1,u)

print("Gray = currently not being sorted; blue = pivot; red = swap")

A = np.array([4,6,1,5,7,3,8,2])
n = 16
A = np.random.randint(0, n**2, n)

clr = np.zeros(A.size)
clr[:] = -1
pretty_print(A,clr)
l = 0
u = A.size

quicksort(A,l,u)

for i in range(A.size-1):
    assert(A[i] <= A[i+1])
