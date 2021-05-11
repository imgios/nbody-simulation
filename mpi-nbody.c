#include "mpi.h"
#include <stdio.h>

int main (int argc, int ** argv) {
    int numtasks, rank, tag = 1;
    // TODO: Buffer declaration
    MPI_Status status;

    // Init MPI env
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
}