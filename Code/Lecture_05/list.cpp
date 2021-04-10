#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cassert>
using std::cout;
using std::endl;

struct Node
{
    int data;
    Node *next;
};

void Wait()
{
    struct timespec req, rem;
    req.tv_sec = 0;
    req.tv_nsec = (1 + rand() % 10) * 100000;
    /* Wait time in nanoseconds */
    nanosleep(&req, &rem);
}

void Visit(Node *curr_node)
{
    /* Add 1 to data */
    ++(curr_node->data);
}

void IncrementListItems(Node *head)
{
    #pragma omp parallel
    {
        #pragma omp single
        {
            Node *curr_node = head;

            while (curr_node)
            {
                printf("Master thread. %p\n", (void *)curr_node);
                #pragma omp task
                {
                    // curr_node is firstprivate by default
                    Wait();
                    int tid = omp_get_thread_num();
                    Visit(curr_node);
                    printf("Task @%2d: node %p data %d\n",
                           tid, (void *)curr_node, curr_node->data);
                }
                curr_node = curr_node->next;
            }
        }
    }
}

int main()
{
    Node *root = new Node;

    int size = 10;
    int i;

    // Fill the list
    Node *head = root;
    root->data = 0;

    for (i = 0; i < size - 1; i++)
    {
        head = head->next = new Node;
        head->data = i + 1;
    }

    head->next = NULL;

    head = root;

    while (head != NULL)
    {
        cout << "data = " << head->data << endl;
        head = head->next;
    }

    IncrementListItems(root);

    cout << "Done incrementing data\n";

    head = root;

    i = 0;

    while (head != NULL)
    {
        cout << "data = " << head->data << endl;
        assert(head->data == ++i);
        head = head->next;
    }
}

#if 0

void IncrementListItems(Node* head) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            Node* curr_node = head;
            while(curr_node) {
                #pragma omp task
                {
                    // curr_node is firstprivate by default
                    Visit(curr_node);
                }
                curr_node = curr_node->next;
            }
        }
    }
}

#endif
