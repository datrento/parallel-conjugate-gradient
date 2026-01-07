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

void csr_sparse_matvec_mult_local(const CSR *A_local, const double *x_full, double *y_local)
{
    /**
     * Performs sparse matrix-vector multiplication y_local = A_local * x_full
     * used for local SpMV where the full vector x is available: no halos needed
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

        if (A->inv_diag)
        {
            free(A->inv_diag);
            A->inv_diag = NULL;
        }
    }

    // Reset CSR structure
    A->n = 0;
    A->n_local = 0;
    A->nnz = 0;
    A->row_start = 0;
    A->inv_diag = NULL;
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

void csr_sparse_matvec_mult_internal(CSR *A, Grid3D *G, double *x, double *y)
{
    /**
     * Performs sparse matrix-vector multiplication y = A * x
     * Handles only the internal rows that do not require halo regions.
     *
     * Parameters:
     * - A: Local CSR matrix
     * - G: Local grid information
     * - x: Local input vector
     * - y: Output vector
     * Returns: void
     **/

    int plane_size = G->nx * G->ny;
    // Skip the first plane and the last plane
    for (int i = plane_size; i < A->n_local - plane_size; i++)
    {
        double sum = 0.0;
        for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++)
        {
            sum += A->values[j] * x[A->col_indices[j] - A->row_start];
        }
        y[i] = sum;
    }
}

static void compute_row(CSR *A, int i, int plane_size, double *x, double *halo_up, double *halo_down, double *y)
{
    /**
     * Helper function to compute a single row of the SpMV considering halo regions.
     * Parameters:
     * - A: Local CSR matrix
     * - i: Row index to compute
     * - plane_size: Size of one xy-plane
     * - x: Local input vector
     * - halo_up: Halo buffer from upper neighbor
     * - halo_down: Halo buffer from lower neighbor
     * - y: Output vector
     * Returns: void
     **/
    int n_local = A->n_local;
    int global_row_start = A->row_start;
    int global_row_end = global_row_start + n_local;

    double sum = 0.0;
    for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++)
    {
        int col = A->col_indices[j];
        double val = A->values[j];
        if (col >= global_row_start && col < global_row_end)
            // Case 1: The data is local to this process
            sum += val * x[col - global_row_start];
        else if (col < global_row_start)
            // Case 2: Data comes from the lower neighbor (halo_down)
            // col is in range [global_row_start - plane_size, global_row_start - 1]
            sum += val * halo_down[col - (global_row_start - plane_size)];
        else
            // Case 3: Data comes from the upper neighbor (halo_up)
            // col is in range [global_row_end, global_row_end + plane_size - 1]
            sum += val * halo_up[col - global_row_end];
    }
    y[i] = sum;
}

void csr_sparse_matvec_mult_boundary(CSR *A, Grid3D *G, double *x, double *halo_up, double *halo_down, double *y)
{
    /**
     * Performs sparse matrix-vector multiplication y = A * x
     * Handles only the boundary rows that require halo regions.
     *
     * Parameters:
     * - A: Local CSR matrix
     * - G: Local grid information
     * - x: Local input vector
     * - halo_up: Halo buffer from upper neighbor
     * - halo_down: Halo buffer from lower neighbor
     * - y: Output vector
     * Returns: void
     **/

    int n_local = A->n_local;
    int plane_size = G->nx * G->ny;

    // Process bottom boundary
    int end_bottom = (plane_size < n_local) ? plane_size : n_local;
    for (int i = 0; i < end_bottom; i++)
    {
        compute_row(A, i, plane_size, x, halo_up, halo_down, y); // helper or inline logic
    }

    // Process top boundary
    int start_top = n_local - plane_size;
    if (start_top < end_bottom)
        start_top = end_bottom; // Avoid overlap

    for (int i = start_top; i < n_local; i++)
    {
        compute_row(A, i, plane_size, x, halo_up, halo_down, y);
    }
}