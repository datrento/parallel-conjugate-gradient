#pragma once
#include <mpi.h>
#include <stddef.h>

typedef struct
{
    int n;            // Global Number of rows
    int n_local;      // Number of local rows
    int row_start;    // Starting row index in global matrix per process
    int nnz;          // Number of non-zero entries (local)
    double *values;   // Non-zero values
    int *col_indices; // Column indices of non-zero values
    int *row_ptr;     // Row pointer
} CSR;

// Rank 0 builds the full SPD matrix of size grid_size^3 x grid_size^3 in CSR format
int csr_build_spd_full(CSR *A, int grid_size, int rank);

// Distributes the CSR matrix A from rank 0 to all processes, each receiving its local part in A_local
int csr_distribute(CSR *A, CSR *A_local, int n, int n_local, int row_start, int rank, int size, MPI_Comm comm);

// SpMV: y_local = A_local * x
void csr_sparse_matvec_mult_local(const CSR *A_local, const double *x_full, double *y_local);

// Extracts the diagonal elements from the local CSR matrix A_local.
void csr_get_diagonal_local(const CSR *A_local, double *diag_A_local, int rank);

void print_csr_memory_usage(int rank, size_t nnz, int n_local, const char *label);

// Free CSR matrix
void csr_free(CSR *A);

// Export CSR matrix to Matrix Market format for verification
void export_csr_to_mtx(const char *filename, CSR *A, int n, int rank);

// Check spd properties of the matrix
void verify_spd_properties(CSR *A, int rank);