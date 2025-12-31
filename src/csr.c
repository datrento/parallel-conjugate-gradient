#include "csr.h"
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <limits.h>
#include <stddef.h>
void verify_spd_properties(CSR *A, int rank)
{
    /***
     * Verifies the symmetric positive definite (SPD) properties of the CSR matrix A.
     * Checks for symmetry and diagonal dominance.
     * Parameters:
     *     A: pointer to CSR matrix
     *     rank: MPI rank for logging purposes
     * Returns: void
     * **/
    int n = A->n;

    // Check diagonal dominance and symmetry
    for (int i = 0; i < n; i++)
    {
        double diag_val = 0.0;
        double off_diag_sum = 0.0;

        for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++)
        {
            int col = A->col_indices[j];
            double val = A->values[j];

            if (col == i)
            {
                diag_val = val;
            }
            else
            {
                off_diag_sum += fabs(val);
            }

            // Check symmetry:  A(i,j) should equal A(j,i)
            // Find A(col, i)
            int found = 0;
            for (int k = A->row_ptr[col]; k < A->row_ptr[col + 1]; k++)
            {
                if (A->col_indices[k] == i)
                {
                    if (fabs(A->values[k] - val) > 1e-10)
                    {
                        fprintf(stderr, "[rank %d] Symmetry violated at (%d,%d): %f != %f\n",
                                rank, i, col, val, A->values[k]);
                    }
                    found = 1;
                    break;
                }
            }
            if (!found && col != i)
            {
                fprintf(stderr, "[rank %d] Symmetry violated:  (%d,%d) exists but (%d,%d) missing\n",
                        rank, i, col, col, i);
                fflush(stderr);
            }
        }

        // Check diagonal dominance
        if (diag_val <= off_diag_sum)
        {
            fprintf(stderr, "[rank %d] Warning: Row %d not strictly diagonally dominant:  diag=%f, sum=%f\n",
                    rank, i, diag_val, off_diag_sum);
            fflush(stderr);
            break;
        }
    }

    printf("[rank %d] SPD with diagonal dominance verification complete\n", rank);
    fflush(stdout);
}
void csr_get_diagonal_local(const CSR *A_local, double *diag_A_local, int rank)
{
    /**
     * Extracts the diagonal elements from the local CSR matrix A_local.
     * Parameters:
     *     A_local: local CSR matrix
     *     diag_A_local: output array to store diagonal elements (size n_local)
     *     rank: MPI rank for logging purposes
     * Returns: void
     */
    for (int r = 0; r < A_local->n_local; r++)
    {
        int global_row = A_local->row_start + r;

        double diag_value = 0.0;

        // diagonal entry (where col_index == global_row)
        for (int j = A_local->row_ptr[r]; j < A_local->row_ptr[r + 1]; j++)
        {
            if (A_local->col_indices[j] == global_row)
            {
                diag_value = A_local->values[j];
                break;
            }
        }

        if (diag_value == 0.0)
        {
            fprintf(stderr, "[rank %d]Warning: Zero diagonal entry found at global row %d\n", rank, global_row);
            fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); // since it's called by multiple places
        }

        diag_A_local[r] = diag_value;
    }
}

void csr_sparse_matvec_mult_local(const CSR *A_local, const double *x_full, double *y_local)
{
    /**
     * Performs sparse matrix-vector multiplication y_local = A_local * x_full
     * Parameters:
     *      A_local: local CSR matrix
     *      x_full: full input vector (size n)
     *      y_local: output vector (size n_local)
     * Returns: void
     */
    for (int i = 0; i < A_local->n_local; i++)
    {
        double sum = 0.0;
        for (int j = A_local->row_ptr[i]; j < A_local->row_ptr[i + 1]; j++)
        {
            sum += A_local->values[j] * x_full[A_local->col_indices[j]];
        }
        y_local[i] = sum;
    }
}

void print_csr_memory_usage(int rank, size_t nnz, int n, const char *label)
{
    /***
     * Prints the memory usage of a CSR matrix given its number of non-zero entries (nnz).
     * Parameters:
     *      rank - MPI rank for logging purposes
     *      nnz - number of non-zero entries
     *      n - number of rows/columns (square matrix)
     *      label - label to identify the matrix (e.g., "local", "full")
     * Returns: void
     *
     ***/
    size_t int_size = sizeof(int);
    size_t double_size = sizeof(double);

    size_t row_ptr_size = (n + 1) * int_size;
    size_t col_indices_size = nnz * int_size;
    size_t values_size = nnz * double_size;

    double mb = 1024.0 * 1024.0;
    size_t total_size = row_ptr_size + col_indices_size + values_size;

    // dense total size
    size_t dense_size = (size_t)n * n * double_size;
    // compare it with the denser matrix memory usage use TB for the full dense matrix
    printf("[rank %d] CSR Matrix (%s): nnz=%zu, n=%d\n", rank, label, nnz, n);
    printf("[rank %d] %s: values_mem=%.2f MB col_indices_mem=%.2f MB row_ptr_mem=%.2f MB total_mem=%.2f GB dense_equiv_mem=%.2f TB\n",
           rank, label, (double)values_size / mb, (double)col_indices_size / mb,
           (double)row_ptr_size / mb, (double)total_size / (mb * 1024),
           (double)dense_size / (mb * 1024 * 1024));
    fflush(stdout);
}

int csr_distribute(
    CSR *A_full, CSR *A_local, int n,
    int n_local, int row_start,
    int rank, int size, MPI_Comm comm)
{
    /***
     * Distributes the CSR matrix A_full from rank 0 to all processes,
     * each receiving its local part in A_local.
     *
     * Parameters:
     *      A_full: Pointer to full CSR matrix on rank 0
     *      A_local: Pointer to local CSR matrix to be filled
     *      n: Global matrix size
     *      n_local: Number of local rows for this process
     *      row_start: Starting row index in global matrix for this process
     *      rank: MPI rank
     *      size: Number of MPI processes
     *      comm: MPI communicator
     *
     * Returns:
     *      0 on success, -1 on failure
     *
     ***/

    A_local->n = n;

    // Compute row distribution for All ranks
    int base = n / size;
    int remainder = n % size;
    A_local->n_local = n_local;
    A_local->row_start = row_start; // starting row index in global matrix

    // Step 1: Scatter row_ptr segments
    int *local_row_ptr_temp = (int *)malloc((A_local->n_local + 1) * sizeof(int));

    if (!local_row_ptr_temp)
    {
        fprintf(stderr, "[rank %d] Error allocating memory for local_row_ptr_temp\n", rank);
        fflush(stderr);
        return -1;
    }

    if (rank == 0)
    {
        // Prepare sendcounts and displs for row_ptr
        int *sendcounts = (int *)malloc(size * sizeof(int));
        int *displs = (int *)malloc(size * sizeof(int));

        if (!sendcounts || !displs)
        {
            fprintf(stderr, "[rank %d] Error allocating memory for sendcounts or displs\n", rank);
            fflush(stderr);
            free(local_row_ptr_temp);
            return -1;
        }

        for (int i = 0; i < size; i++)
        {
            int start = i * base + (i < remainder ? i : remainder);
            int count = ((i < remainder) ? (base + 1) : base);
            sendcounts[i] = count + 1; // +1 for row_ptr
            displs[i] = start;
        }

        // Distribute A_full->row_ptr to all ranks' local_row_ptr_temp
        MPI_Scatterv(A_full->row_ptr, sendcounts, displs, MPI_INT,
                     local_row_ptr_temp, A_local->n_local + 1, MPI_INT,
                     0, comm);

        // Free allocation arrays
        free(sendcounts);
        free(displs);
    }
    else
    {
        // Receive row_ptr segments
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT,
                     local_row_ptr_temp, A_local->n_local + 1, MPI_INT,
                     0, comm);
    }

    // Adjust local_row_ptr to start from 0 for each rank and compute local nnz(non zeros)
    int nnz_offset = local_row_ptr_temp[0];
    A_local->nnz = local_row_ptr_temp[A_local->n_local] - nnz_offset;

    // Allocate A_local->row_ptr to store adjusted row_ptr
    A_local->row_ptr = (int *)malloc((A_local->n_local + 1) * sizeof(int));

    // Check allocation success
    if (!A_local->row_ptr)
    {
        fprintf(stderr, "[rank %d] Error allocating memory for local row_ptr\n", rank);
        free(local_row_ptr_temp);
        return -1;
    }

    // Adjust A_local->row_ptr to start from 0 for each rank
    for (int i = 0; i < A_local->n_local + 1; i++)
    {
        A_local->row_ptr[i] = local_row_ptr_temp[i] - nnz_offset;
    }

    // Free temporary local_row_ptr storage
    free(local_row_ptr_temp);

    // Step 2: Scatter col_indices and values based on each rank's nnz
    // allocate col_indices and values for A_local
    A_local->col_indices = (int *)malloc(A_local->nnz * sizeof(int));
    A_local->values = (double *)malloc(A_local->nnz * sizeof(double));

    if (!A_local->col_indices)
    {
        fprintf(stderr, "[rank %d] malloc failed for col_indices (need %d ints = %.2f MB)\n",
                rank, A_local->nnz, A_local->nnz * sizeof(int) / (1024.0 * 1024.0));
        free(A_local->row_ptr);
        return -1;
    }

    if (!A_local->values)
    {
        fprintf(stderr, "[rank %d] malloc failed for values (need %d doubles = %.2f MB)\n",
                rank, A_local->nnz, A_local->nnz * sizeof(double) / (1024.0 * 1024.0));
        free(A_local->row_ptr);
        free(A_local->col_indices);
        return -1;
    }

    // Print memory usage for local CSR matrix per rank
    print_csr_memory_usage(rank, A_local->nnz, A_local->n_local, "local");

    if (rank == 0)
    {
        // Prepare sendcounts and displs for col_indices and values
        int *sendcounts_nnz = (int *)malloc(size * sizeof(int));
        int *displs_nnz = (int *)malloc(size * sizeof(int));

        if (!sendcounts_nnz || !displs_nnz)
        {
            fprintf(stderr, "[rank %d] Error allocating memory for sendcounts_nnz or displs_nnz\n", rank);
            fflush(stderr);
            free(A_local->row_ptr);
            free(A_local->col_indices);
            free(A_local->values);
            return -1;
        }

        for (int i = 0; i < size; i++)
        {
            // let say remainder = 2, base = 3, size = 5
            // p0_start = 0 * 3 + 0 = 0,    count = 4 => sendcounts_nnz[0] = 4
            // p1_start = 1 * 3 + 1 = 4,    count = 4 => sendcounts_nnz[1] = 4
            // p2_start = 2 * 3 + 2 = 8,    count = 3 => sendcounts_nnz[2] = 3
            // p3_start = 3 * 3 + 2 = 11,   count = 3 => sendcounts_nnz[3] = 3
            // p4_start = 4 * 3 + 2 = 14,   count = 3 => sendcounts_nnz[4] = 3
            int start = i * base + (i < remainder ? i : remainder);
            int count = ((i < remainder) ? (base + 1) : base);
            sendcounts_nnz[i] = A_full->row_ptr[start + count] - A_full->row_ptr[start]; // start to end nnz for this rank
            displs_nnz[i] = A_full->row_ptr[start];
        }

        // Distribute col_indices and values to all ranks' A_local
        MPI_Scatterv(A_full->col_indices, sendcounts_nnz, displs_nnz, MPI_INT,
                     A_local->col_indices, A_local->nnz, MPI_INT,
                     0, comm);

        MPI_Scatterv(A_full->values, sendcounts_nnz, displs_nnz, MPI_DOUBLE,
                     A_local->values, A_local->nnz, MPI_DOUBLE,
                     0, comm);

        // Free allocation arrays
        free(sendcounts_nnz);
        free(displs_nnz);
    }
    else
    {
        // Receive col_indices and values
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT,
                     A_local->col_indices, A_local->nnz, MPI_INT,
                     0, comm);

        MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE,
                     A_local->values, A_local->nnz, MPI_DOUBLE,
                     0, comm);
    }

    return 0;
}

void csr_free(CSR *A)
{
    /***
     * Frees the memory allocated for the CSR matrix A.
     * Parameters:
     *      A: Pointer to CSR structure to be freed
     * Returns: void
     ***/
    if (A)
    {
        if (A->row_ptr)
        {
            free(A->row_ptr);
            A->row_ptr = NULL;
        }
        if (A->col_indices)
        {
            free(A->col_indices);
            A->col_indices = NULL;
        }

        if (A->values)
        {
            free(A->values);
            A->values = NULL;
        }
    }

    // Reset CSR structure
    A->n = 0;
    A->n_local = 0;
    A->nnz = 0;
    A->row_start = 0;
}

void export_csr_to_mtx(const char *filename, CSR *A, int n, int rank)
{
    /***
     * Exports a CSR matrix A to a Matrix Market (.mtx) file.
     * so it would be easy to read in python and verify correctness of SPD matrix properties
     * Parameters:
     *      filename: output file name
     *      A: pointer to CSR matrix
     *      n: global matrix size
     *      rank: MPI rank for logging purposes
     * Returns: void
     ***/

    FILE *f = fopen(filename, "w");
    if (!f)
    {
        fprintf(stderr, "[rank %d]Error opening %s\n", rank, filename);
        fflush(stderr);
        return;
    }

    // Write Matrix Market header (general format since i store both (i,j) and (j,i) explicitly)
    fprintf(f, "%%MatrixMarket matrix coordinate real general\n");
    fprintf(f, "%d %d %d\n", n, n, A->nnz);

    // Write non-zero entries
    for (int i = 0; i < n; i++)
    {
        for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++)
        {
            int col = A->col_indices[j];
            double val = A->values[j];
            // Matrix Market format uses 1-based indexing
            fprintf(f, "%d %d %.16e\n", i + 1, col + 1, val);
        }
    }
    fclose(f);

    fprintf(stdout, "[rank %d]Matrix exported to %s in Matrix Market format.\n", rank, filename);
    fflush(stdout);
}
