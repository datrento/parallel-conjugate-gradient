#include "solver_utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void verify_solution(double *Ax_local, double *b_local, int n_local, int rank, MPI_Comm comm)
{
    /***
     * Verifies the solution by computing Ax and comparing it to b.
     * Note: Ax buffer will be overwritten with A*x computation.
     * Parameters:
     *      Ax_local: local buffer to store A*x result
     *      b_local: local right-hand side vector
     *      n_local: number of local rows
     *      rank: MPI rank
     *      comm: MPI communicator
     * Returns: void
     ***/
    // Compute Ax_local = A_local * x
    double local_err2 = 0.0, local_b2 = 0.0;
    for (int i = 0; i < n_local; i++)
    {
        double d = Ax_local[i] - b_local[i];
        local_err2 += d * d;
        local_b2 += b_local[i] * b_local[i];
    }
    double err2 = 0.0, b2 = 0.0;
    double sums[2] = {local_err2, local_b2};
    double result[2] = {0.0, 0.0};
    MPI_Reduce(sums, result, 2, MPI_DOUBLE, MPI_SUM, 0, comm);
    err2 = result[0];
    b2 = result[1];

    if (rank == 0)
    {
        printf("[rank %d]Verification: ||Ax - b|| = %.6e, ||Ax - b|| / ||b|| (relative) = %.6e\n",
               rank, sqrt(err2), sqrt(err2) / sqrt(b2));
    }
}

void halo_exchange_init(Grid3D *G, double *v_local, double *r_up, double *r_down, MPI_Request *reqs)
{
    int plane_size = G->nx * G->ny;

    // Initialize requests to NULL so Waitall works even if some are unused
    for (int i = 0; i < 4; i++)
        reqs[i] = MPI_REQUEST_NULL;

    if (G->up != MPI_PROC_NULL)
    {
        // Tag 1: Receiving from above. Tag 0: Sending to above.
        MPI_Irecv(r_up, plane_size, MPI_DOUBLE, G->up, 1, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(&v_local[(G->local_nz - 1) * plane_size], plane_size, MPI_DOUBLE, G->up, 0, MPI_COMM_WORLD, &reqs[2]);
    }

    if (G->down != MPI_PROC_NULL)
    {
        // Tag 0: Receiving from below. Tag 1: Sending to below.
        MPI_Irecv(r_down, plane_size, MPI_DOUBLE, G->down, 0, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(v_local, plane_size, MPI_DOUBLE, G->down, 1, MPI_COMM_WORLD, &reqs[3]);
    }
}