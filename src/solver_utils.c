/*
 * Utilities for splitting Ap = A*p into local and remote contributions.
 * Notes:
 * - csr_local_Ap_contribution uses p_local with [row_start, row_end) indexing.
 * - csr_global_Ap_contribution adds remote terms using global p.
 * - check_the_ap_computation_is_valid runs once and allocates a temporary buffer.
 */
#include "solver_utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void csr_local_Ap_contribution(
    const CSR *A_local,
    const double *p_local,
    double *Ap_local,
    int n_local,
    int row_start,
    int row_end)
{
    /***
     * Computes the local contribution to Ap_local using only the local part of p.
     * Parameters:
     *      A_local: local CSR matrix
     *      p_local: local part of vector p
     *      Ap_local: local A*p vector to be computed
     *      n_local: number of local rows
     *      row_start: starting global row index for this local matrix
     *      row_end: ending global row index for this local matrix
     * Returns: void
     ***/
    for (int i = 0; i < n_local; i++)
    {
        double sum = 0.0;

        for (int j = A_local->row_ptr[i]; j < A_local->row_ptr[i + 1]; j++)
        {
            int col = A_local->col_indices[j];

            // check if col index belongs to local part of p (for each value of A_local row check if the corresponding p is local)
            if (col >= row_start && col < row_end)
            {
                // local part of p
                sum += A_local->values[j] * p_local[col - row_start]; // adjust index for local p
            }
        }
        Ap_local[i] = sum;
    }
}

void csr_global_Ap_contribution(
    const CSR *A_local,
    const double *p,
    double *Ap_local,
    int n_local,
    int row_start,
    int row_end)
{
    /***
     * Completes the Ap_local computation by adding contributions from the remote parts of p.
     * Parameters:
     *      A_local: local CSR matrix
     *      p: global vector
     *      Ap_local: local A*p vector to be updated
     *      n_local: number of local rows
     *      row_start: starting global row index for this local matrix
     *      row_end: ending global row index for this local matrix
     * Returns: void
     *
     ***/
    for (int i = 0; i < n_local; i++)
    {
        double sum = Ap_local[i];
        for (int j = A_local->row_ptr[i]; j < A_local->row_ptr[i + 1]; j++)
        {
            int col = A_local->col_indices[j];
            // check if col index belongs to remote part of p
            if (col < row_start || col >= row_end)
            {
                // remote part of p
                sum += A_local->values[j] * p[col];
            }
        }
        Ap_local[i] = sum;
    }
}

void check_the_ap_computation_is_valid(const CSR *A_local, double *Ap_local, double *p, int n_local, int rank)
{
    /***
     * Debug function to verify that the split computation of Ap_local
     * using local and global contributions matches the direct computation.
     *
     * Parameters:
     *      A_local: local CSR matrix
     *      Ap_local: computed A*p local vector
     *      p: global vector
     *      n_local: number of local rows
     *      rank: MPI rank for logging purposes
     *
     * Returns: void
     ***/
    static int checked_once = 0;
    if (!checked_once)
    {
        double *Ap_ref = (double *)malloc(n_local * sizeof(double));
        csr_sparse_matvec_mult_local(A_local, p, Ap_ref);

        double maxdiff = 0.0;

        for (int i = 0; i < n_local; i++)
        {
            double diff = fabs(Ap_local[i] - Ap_ref[i]);
            if (diff > maxdiff)
                maxdiff = diff;
        }

        if (rank == 0)
        {
            printf("[rank %d][DEBUG] Ap split vs max diff: %.6e\n", rank, maxdiff);
            fflush(stdout);
        }

        free(Ap_ref);
        checked_once = 1;
    }
}

void verify_solution(CSR *A_local, double *x, double *b_local, double *Ax_local, int n_local, int rank, MPI_Comm comm)
{
    /***
     * Verifies the solution by computing Ax and comparing it to b.
     * Note: Ax buffer will be overwritten with A*x computation.
     * Parameters:
     *      A_local: local CSR matrix
     *      x: global solution vector
     *      b_local: local right-hand side vector
     *      Ax_local: local buffer to store A*x result
     *      n_local: number of local rows
     *      rank: MPI rank
     *      comm: MPI communicator
     * Returns: void
     ***/
    csr_sparse_matvec_mult_local(A_local, x, Ax_local);

    double local_err2 = 0.0, local_b2 = 0.0;
    for (int i = 0; i < n_local; i++)
    {
        double d = Ax_local[i] - b_local[i];
        local_err2 += d * d;
        local_b2 += b_local[i] * b_local[i];
    }
    double err2 = 0.0, b2 = 0.0;
    MPI_Reduce(&local_err2, &err2, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_b2, &b2, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    if (rank == 0)
    {
        printf("[rank %d]Verification: ||Ax - b|| = %.6e, ||Ax - b|| / ||b|| (relative) = %.6e\n",
               rank, sqrt(err2), sqrt(err2) / sqrt(b2));
    }
}