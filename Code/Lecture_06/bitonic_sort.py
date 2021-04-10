import numpy as np
import math as m

up = 1
down = 0


def pretty_print(a, clr):
    print('[', end=' ')
    for i in range(a.size):
        if clr[i] == 1:
            print('\033[91m', a[i], end=' ')
        elif clr[i] == 2:
            print('\033[92m', a[i], end=' ')
        elif clr[i] == 3:
            print('\033[93m', a[i], end=' ')
        elif clr[i] == 4:
            print('\033[34m', a[i], end=' ')
        else:
            print('\033[0m', a[i], end=' ')
    print('\033[0m', ']')


def bitonic_sort(a, start, end, flag):
    if end <= start+1:
        return

    length = end - start

    if length % 2 != 0:
        print("The length of a (sub)sequence is not divisible by 2")
        exit

    split_length = length >> 1

    # Bitonic compare
    for i in range(start, start+split_length):
        if flag == up:
            if a[i] > a[i+split_length]:
                a[i], a[i+split_length] = a[i+split_length], a[i]
        else:
            if a[i] < a[i+split_length]:
                a[i], a[i+split_length] = a[i+split_length], a[i]

    # This piece of code is just for the pretty printing
    color = np.zeros(n)
    power_of_two = 1
    while ((1 << power_of_two) != length):
        power_of_two += 1
    color[start:start+length] = 1+power_of_two % 4
    pretty_print(a, color)

    # Recursive calls
    bitonic_sort(a, start, start+split_length, flag)
    bitonic_sort(a, start+split_length, end, flag)


n = 1 << 4
a = np.random.randint(0, n**2, n)
b = a.copy()

i = 2
while i <= n:
    flag = up
    print('')
    color = np.zeros(n)
    pretty_print(a, color)
    for j in range(0, n, i):
        # Sorting sequences of increasing length
        bitonic_sort(a, j, j+i, flag)
        flag = 1-flag
    i = i << 1

b = np.sort(b)
assert(np.array_equal(a, b))
