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

// Helper function to find which process owns a column
static int find_column_owner(int col, int size, int n)
{
    int base = n / size;
    int remainder = n % size;

    // Same distribution logic as row_start calculation
    if (col < (base + 1) * remainder)
    {
        return col / (base + 1);
    }
    else
    {
        return remainder + (col - (base + 1) * remainder) / base;
    }
}

void analyze_communication_pattern(
    const CSR *A_local,
    int n_local,
    int row_start,
    int row_end,
    int rank,
    int size,
    int n)
{
    /***
     * Analyzes the communication pattern for SpMV operation.
     * Prints how many remote columns are needed and from which processes.
     *
     * Parameters:
     *      A_local: local CSR matrix
     *      n_local: number of local rows
     *      row_start: starting global row index
     *      row_end: ending global row index
     *      rank:  MPI rank
     *      size:  number of MPI processes
     *      n: global matrix size
     * Returns:  void
     ***/

    // Count remote columns needed
    int total_remote_cols = 0;
    int total_local_cols = 0;

    // Count how many values needed from each process
    int *cols_from_process = (int *)calloc(size, sizeof(int));

    for (int i = 0; i < n_local; i++)
    {
        for (int j = A_local->row_ptr[i]; j < A_local->row_ptr[i + 1]; j++)
        {
            int col = A_local->col_indices[j];

            if (col >= row_start && col < row_end)
            {
                // Local column
                total_local_cols++;
            }
            else
            {
                // Remote column
                total_remote_cols++;

                // Determine which process owns this column
                int owner = find_column_owner(col, size, n);
                cols_from_process[owner]++;
            }
        }
    }

    // Print analysis
    printf("[rank %d] === Communication Analysis ===\n", rank);
    printf("[rank %d] Local columns needed:    %d\n", rank, total_local_cols);
    printf("[rank %d] Remote columns needed: %d\n", rank, total_remote_cols);
    printf("[rank %d] Total non-zeros: %d\n", rank, A_local->nnz);
    printf("[rank %d] Remote percentage: %.2f%%\n", rank,
           100.0 * total_remote_cols / A_local->nnz);

    printf("[rank %d] Columns needed from each process:\n", rank);
    for (int p = 0; p < size; p++)
    {
        if (cols_from_process[p] > 0)
        {
            printf("[rank %d]   Process %d:  %d values\n", rank, p, cols_from_process[p]);
        }
    }

    // Compare to Allgatherv
    int allgatherv_receives = n - n_local;
    printf("[rank %d] Allgatherv would receive: %d values\n", rank, allgatherv_receives);
    printf("[rank %d] Halo would receive: %d values\n", rank, total_remote_cols);

    if (total_remote_cols < allgatherv_receives)
    {
        printf("[rank %d] => Halo exchange would SAVE %.2f%% communication\n", rank, 100.0 * (allgatherv_receives - total_remote_cols) / allgatherv_receives);
    }
    else
    {
        printf("[rank %d] => Allgatherv is better (or similar)\n", rank);
    }
    printf("[rank %d] =============================\n", rank);
    fflush(stdout);

    free(cols_from_process);
}

// Add this function at the end of solver_utils.c

void csr_spmv_halo(
    const CSR *A_local,
    const double *p_local,
    double *Ap_local,
    const HaloExchange *halo,
    int n_local,
    int row_start,
    int row_end)
{
    // Initialize Ap_local to zero
    for (int i = 0; i < n_local; i++)
    {
        Ap_local[i] = 0.0;
    }

    // Check if halo is valid
    if (halo->total_ghost_size > 0 && (halo->ghost_values == NULL || halo->ghost_to_global == NULL))
    {
        fprintf(stderr, "ERROR: Invalid halo structure - ghost arrays are NULL!\n");
        return;
    }

    // Perform SpMV using both local and ghost values
    for (int i = 0; i < n_local; i++)
    {
        double sum = 0.0;

        for (int j = A_local->row_ptr[i]; j < A_local->row_ptr[i + 1]; j++)
        {
            int col = A_local->col_indices[j];
            double val = A_local->values[j];

            if (col >= row_start && col < row_end)
            {
                // Local column - use p_local
                int local_col = col - row_start;
                sum += val * p_local[local_col];
            }
            else
            {
                // Remote column - use ghost values
                // Find this column in ghost_to_global mapping
                int ghost_idx = -1;
                for (int k = 0; k < halo->total_ghost_size; k++)
                {
                    if (halo->ghost_to_global[k] == col)
                    {
                        ghost_idx = k;
                        break;
                    }
                }

                if (ghost_idx == -1)
                {
                    fprintf(stderr, "ERROR: Column %d not found in ghost values (row %d, local row %d)!\n",
                            col, row_start + i, i);
                    fprintf(stderr, "       Ghost size: %d, row range: [%d, %d)\n",
                            halo->total_ghost_size, row_start, row_end);
                    // Print what's in ghost_to_global
                    fprintf(stderr, "       Ghost columns: ");
                    for (int k = 0; k < halo->total_ghost_size && k < 10; k++)
                    {
                        fprintf(stderr, "%d ", halo->ghost_to_global[k]);
                    }
                    fprintf(stderr, "\n");
                    continue;
                }

                sum += val * halo->ghost_values[ghost_idx];
            }
        }

        Ap_local[i] = sum;
    }
}