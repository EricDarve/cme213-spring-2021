#include <thread>
#include <future>
#include <chrono>
#include <iostream>
#include <cassert>
#include <vector>

using namespace std;

void f1()
{
    cout << "f1() called\n";
}

void f2(int m)
{
    /* optional: make thread wait a bit */
    this_thread::sleep_for(chrono::milliseconds(10));
    cout << "f2() called with m = " << m << endl;
}

void f3(int &k)
{
    this_thread::sleep_for(chrono::milliseconds(20));
    cout << "f3() called; k is passed by reference; k = " << k << endl;
    k += 3;
}

void f4(int m, int &k)
{
    // todo
}

void accumulate(vector<int>::iterator first,
                vector<int>::iterator last,
                promise<int> accumulate_promise)
{
    int sum = 0;
    auto it = first;
    for (; it != last; ++it)
        sum += *it;
    accumulate_promise.set_value(sum); // Notify future
}

void get_max(vector<int>::iterator first,
             vector<int>::iterator last,
             promise<int> max_promise)
{
    int sum = *first;
    auto it = first;
    for (; it != last; ++it)
        sum = (*it > sum ? *it : sum);
    // todo
    // max_promise
}

int main(void)
{
    // Demonstrate using thread constructor
    thread t1(f1);

    int m = 5;
    // With an argument
    thread t2(f2, m);

    int k = 7;
    // With a reference
    thread t3(f3, ref(k)); /* use ref to pass a reference */

    /* wait for all threads to finish */
    t1.join();
    t2.join();
    t3.join();

    cout << "k is now equal to " << k << endl;
    assert(k == 10);

    // Exercise
    // f4(m,k) should compute k += m
    // Before f4(m,k):
    // m = 5
    // k = 10
    // After:
    // k = 15
    // todo: thread t4(...)
    thread t4;
    // todo
    // t4.join();

    cout << "k is now equal to " << k << endl;
    assert(k == 15);

    // Demonstrate using promise<int> to return a value
    vector<int> vec_1 = {1, 2, 3, 4, 5, 6};
    promise<int> accumulate_promise; // Will store the int
    future<int> accumulate_future = accumulate_promise.get_future();
    // Used to retrieve the value asynchronously, at a later time

    thread t5(accumulate, vec_1.begin(), vec_1.end(),
              move(accumulate_promise));
    // move() will "move" the resources allocated for accumulate_promise

    // future::get() waits until the future has a valid result and retrieves it
    cout << "result of accumulate_future [21 expected] = " << accumulate_future.get() << '\n';
    t5.join();
    // Wait for thread completion

    vector<int> vec_2 = {1, -2, 4, -10, 5, 4};
    // todo
    // max_promise;
    // max_future = ...

    // todo
    thread t6;
    int max_result = 0;

    // todo
    // max_result = ...
    cout << "result of max_future [5 expected] = " << max_result << '\n';
    assert(max_result == 5);

    t6.join();
    // Wait for thread completion

    return 0;
}