#include <mpi.h>

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
  // Initialize the MPI environment
  MPI_Init(NULL, NULL);

  // Find out the rank and size
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // We need at least 2 processes
  if (world_size < 2) {
    fprintf(stderr, "World size must be greater than 1 for %s\n", argv[0]);
    // Use this command to terminate MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  srand(rank + 10);  // Initialize the random number generator
  int number_send, number_recv;
  number_send = rand() % 100;

  // Receive from the lower process and send to the higher process.
  int rank_receiver = rank == world_size - 1 ? 0 : rank + 1;
  MPI_Send(&number_send, 1, MPI_INT, rank_receiver, 0, MPI_COMM_WORLD);
  printf("Process %d sent \t\t %2d to   process %d\n", rank, number_send,
         rank_receiver);

  int rank_sender = rank == 0 ? world_size - 1 : rank - 1;
  MPI_Recv(&number_recv, 1, MPI_INT, rank_sender, 0, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  printf("Process %d received \t %2d from process %d\n", rank, number_recv,
         rank_sender);

  MPI_Finalize();

  return 0;
}
