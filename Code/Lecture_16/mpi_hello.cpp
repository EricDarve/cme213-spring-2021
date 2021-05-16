#include "mpi.h"
#include <cstdio>
#include <cstdlib>

#define MASTER 0

int main(int argc, char *argv[])
{

  // Some MPI magic to get started
  MPI_Init(&argc, &argv);

  // How many processes are running
  int numprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  // What's my rank?
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // Which node am I running on?
  int len;
  char hostname[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(hostname, &len);

  printf("Hello from rank %2d running on node: %s\n", rank, hostname);

  // Only one processor will do this
  if (rank == MASTER)
  {
    printf("MASTER process: the number of MPI processes is: %2d\n", numprocs);
  }

  // Close down all MPI magic
  MPI_Finalize();

  return 0;
}
