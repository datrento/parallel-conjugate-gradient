/**
 * CSR data structures and sparse kernels used by the solvers.
 * The MPI domain decomposition is along the z-axis (z-slab): each rank owns
 * a contiguous set of xy-planes and exchanges halo planes with its z-neighbors.
 */


#pragma once
#include <mpi.h>
#include <stddef.h>

/**
 * Compressed Sparse Row (CSR) matrix structure with Jacobi preconditioning support.
 * Members:
 *     n: Global number of rows
 *     n_local: Number of local rows
 *     row_start: Global row start index
 *     nnz: Number of non-zero entries (local)
 *     row_ptr: CSR row pointers
 *     col_indices: CSR column indices
 *     values: CSR non-zero values
 *     inv_diag: Inverse of diagonal elements (for Jacobi preconditioning)
 */
typedef struct
{
    int n;            // Global number of rows
    int n_local;      // Number of local rows
    int row_start;    // Global row start index
    int nnz;          // Number of non-zero entries (local)
    int *row_ptr;     // CSR row pointers
    int *col_indices; // CSR column indices
    double *values;   // CSR non-zero values
    double *inv_diag; // Inverse of diagonal elements (for Jacobi preconditioning)
} CSR;

/**
 * Structure to hold 3D grid information for domain decomposition.
 * Members:
 *     nx, ny, nz: Global grid dimensions
 *     local_nz: Local z-dimension size (number of z-planes assigned to this process)
 *     z_start: Starting z-index for this process
 *     up, down: Neighboring process ranks since sliced in z-direction
 */
typedef struct
{
    int nx, ny, nz; // Global grid dimensions
    int local_nz;   // Local z-dimension size (number of z-planes assigned to this process)
    int z_start;    // Starting z-index for this process (global index of the first local z-plane)
    int up, down;   // Neighboring process ranks since sliced in z-direction
} Grid3D;

/**
 * CSR matrix vector multiplication with internal rows only.
 * Parameters:
 *      A: pointer to CSR matrix
 *      G: pointer to Grid3D structure with grid and decomposition info
 *      x: input vector
 *      y: output vector
 * Returns: void
 */
// SpMV: y = A * x for internal rows only (no halos needed)
void csr_sparse_matvec_mult_internal(CSR *A, Grid3D *G, double *x, double *y);

/**
 * CSR matrix vector multiplication with boundary rows only.
 * Parameters:
 *      A: pointer to CSR matrix
 *      G: pointer to Grid3D structure with grid and decomposition info
 *      x: input vector
 *      halo_up: halo data from the upper neighboring process
 *      halo_down: halo data from the lower neighboring process
 *      y: output vector
 * Returns: void
 */
// SpMV: y = A * x for boundary rows only (requires halos)
void csr_sparse_matvec_mult_boundary(CSR *A, Grid3D *G, double *x, double *halo_up, double *halo_down, double *y);

/**
 * CSR matrix vector multiplication for local rows using full input vector.
 * Parameters:
 *      A_local: pointer to local CSR matrix
 *      x_full: full input vector (including halos)
 *      y_local: output vector for local rows
 * Returns: void
 */
// SpMV: y_local = A_local * x full (no halos needed) (used for solution verification and jacobi preconditioned cg)
void csr_sparse_matvec_mult_local(const CSR *A_local, const double *x_full, double *y_local);

/**
 * Print CSR matrix memory usage for debugging.
 * Parameters:
 *      rank: MPI rank for logging purposes
 *      nnz: Number of non-zero entries (local)
 *      n_local: Number of local rows
 *      label: Custom label for the log message
 * Returns: void
 */
void print_csr_memory_usage(int rank, size_t nnz, int n_local, const char *label);

/**
 * Free CSR structure pointers.
 * Parameters:
 *      A: pointer to CSR matrix
 * Returns: void
 */
// Free CSR matrix
void csr_free(CSR *A);

/**
 * Export CSR matrix to Matrix Market format for verification.(used only for smaller grid sizes)
 * Parameters:
 *      filename: output file name
 *      A: pointer to CSR matrix
 *      n: global number of rows
 *      rank: MPI rank for logging purposes
 * Returns: void
 */
// Export CSR matrix to Matrix Market format for verification
void export_csr_to_mtx(const char *filename, CSR *A, int n, int rank);

/**
 * Check if the CSR matrix is symmetric positive definite (SPD).(used only for smaller grid sizes)
 * Parameters:
 *      A: pointer to CSR matrix
 *      rank: MPI rank for logging purposes
 * Returns: void
 */
// Check spd properties of the matrix
void verify_spd_properties(CSR *A, int rank);