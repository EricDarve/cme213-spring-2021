#include <mutex>
#include <vector>
#include <thread>
#include <queue>
#include <string>
#include <iostream>
#include <sstream>

using namespace std;

mutex g_mutex;
queue<string> g_task_queue;

void InsertNewOrder(const string order)
{
    g_task_queue.push(order);
    cout << order << endl;
}

void Delivery()
{
    // Pretend I am really busy doing something here
    // Nice job, isn't it? The pay is good too.
    this_thread::sleep_for(chrono::milliseconds(20));
}

void PizzaDeliveryPronto(int thread_id)
{
    g_mutex.lock();

    while (!g_task_queue.empty())
    {
        printf("Thread %d: %s\n", thread_id, g_task_queue.front().c_str());
        g_task_queue.pop();

        /* At this point, the thread delivers the pizza. */
        /* The mutex is unlocked so that other threads can work. */
        g_mutex.unlock();

        Delivery();

        g_mutex.lock();
    }

    g_mutex.unlock();
    return;
}

#define NTHREADS 4
#define NORDERS 16

int main(int argc, char **argv)
{
    InsertNewOrder("My pizza delivery service");
    InsertNewOrder("Open the store");

    /* Insert some tasks */
    for (int i = 0; i < NORDERS; ++i)
    {
        /* Got a phone call for pizza delivery. */
        stringstream order_no;
        order_no << "Pizza delivery to customer no. " << (rand() % 100 + 1);
        InsertNewOrder(order_no.str());
    }

    InsertNewOrder("Close the store");
    InsertNewOrder("Done. Goodbye CME213.");

    printf("\n\n");

    /* Have threads consume tasks */
    queue<thread> thread_pool;
    for (int i = 0; i < NTHREADS; ++i)
        thread_pool.push(thread(PizzaDeliveryPronto, i));
    // push() or emplace(PizzaDeliveryPronto, i)
    // emplace calls the constructor thread(PizzaDeliveryPronto, i)

    for (int i = 0; i < NTHREADS; ++i)
    {
        thread_pool.front().join();
        thread_pool.pop();
    }

    return 0;
}
