#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
using std::vector;

int main(int argc, char **argv)
{

  const int locn = 5;
  vector<int> localarr(locn);

  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  srand(2017 + (rank << 2));

  for (int i = 0; i < locn; i++)
  {
    localarr[i] = rand() % 10000;
  }

  // This is a trick to print the messages in order
  for (int proc = 0; proc < size; proc++)
  {
    if (rank == proc)
    {
      // This is the turn of process proc to print its message
      printf("Rank %3d has values: ", rank);

      for (int i = 0; i < locn; i++)
      {
        printf(" %5d ", localarr[i]);
      }

      printf("\n");
    }

    if (proc == size - 1 && rank == size - 1)
    {
      printf("\n");
    }

    // A barrier is needed to make sure the messages are printed in order
    MPI_Barrier(MPI_COMM_WORLD);
  }

  int localres[2];
  int globalres[2];
  // Compute the minimum of localarr and store the result in localres[0]
  localres[0] = localarr[0];

  for (int i = 1; i < locn; i++)
    if (localarr[i] < localres[0])
    {
      localres[0] = localarr[i];
    }

  // The second entry is the rank of this process.
  localres[1] = rank;

  /* MPI_Allreduce: the minimum across all processes is computed.
   * The result is broadcast to all processes.
   * MPI_MINLOC: this is like the operator MIN. The difference is that it
   * takes as input two numbers; the first one is used to determine the
   * minimum value.
   * The second number just "goes along for the ride."
   * MPI_2INT: MPI type for 2 integer values. */
  MPI_Allreduce(localres, globalres, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

  /* The difference with MPI_Reduce is that all processes now have the result. */

  if (rank == 0)
  {
    printf("Rank %2d has the lowest value of %d\n\n", globalres[1],
           globalres[0]);
  }

  // For pretty printing
  MPI_Barrier(MPI_COMM_WORLD);

  // Checking that all processes have received the correct value.
  printf("Rank %2d has received the value: %5d\n", rank, globalres[0]);

  MPI_Finalize();

  return 0;
}