
#pragma once
#include "csr.h"
#include <mpi.h>

/**
 * Verifies the solution by computing Ax and comparing it to b.
 * Note: Ax buffer will be overwritten with A*x computation.
 * Parameters:
 *      Ax_local: local buffer to store A*x result
 *      b_local: local right-hand side vector
 *      n_local: number of local rows
 *      rank: MPI rank
 *      comm: MPI communicator
 * Returns: void
 **/
void verify_solution(double *Ax_local, double *b_local,
                     int n_local, int rank, MPI_Comm comm);

/**
 * Initiates non-blocking halo exchanges for the given local vector.
 * Parameters:
 *      G: Grid3D structure with neighbor information
 *      v_local: local vector to exchange halos for
 *      r_up: buffer to receive halo from upper neighbor
 *      r_down: buffer to receive halo from lower neighbor
 *      reqs: array of MPI_Request to hold the requests (size 4)
 * Returns: void
 */
void halo_exchange_init(Grid3D *G, double *v_local, double *r_up, double *r_down, MPI_Request *reqs);