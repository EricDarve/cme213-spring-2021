#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "mpi.h"

#define NPROCS 8

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (nprocs != NPROCS) {
    if (rank == 0) {
      printf("The number of processes must be %d. Terminating.\n", NPROCS);
    }

    MPI_Finalize();  // Don't forget to finalize!
    exit(0);
  }

  /* Extract the original group handle */
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  /* Divide tasks into two distinct groups based upon rank */
  int ranks[2][4] = {{0, 1, 2, 3}, {4, 5, 6, 7}};
  /* These arrays specify the rank to be used
   * to create 2 separate process groups.
   */
  MPI_Group sub_group;
  int mygroup = (rank < NPROCS / 2) ? 0 : 1;
  MPI_Group_incl(world_group, NPROCS / 2, ranks[mygroup], &sub_group);

  /* Create new new communicator and then perform collective communications */
  MPI_Comm sub_group_comm;
  MPI_Comm_create(MPI_COMM_WORLD, sub_group, &sub_group_comm);
  // All processes in that group must call MPI_Comm_create with the same group
  // as argument. This means that MPI_Comm_create should be called by the same
  // processes in the same order. This implies that the set of groups specified
  // across the processes must be disjoint.

  // Summing up the value of the rank for all processes in my group
  int sendbuf = rank;
  int recvbuf;
  MPI_Allreduce(&sendbuf, &recvbuf, 1, MPI_INT, MPI_SUM, sub_group_comm);

  int group_rank;
  MPI_Group_rank(sub_group, &group_rank);
  printf("Rank= %d; Group rank= %d; recvbuf= %d\n", rank, group_rank, recvbuf);

  MPI_Finalize();
}
