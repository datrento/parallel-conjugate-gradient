#include "csr.h"
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

void csr_get_diagonal_local(const CSR *A_local, double *diag_A_local)
{
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
            fprintf(stderr, "Warning: Zero diagonal entry found at global row %d\n", global_row);
        }

        diag_A_local[r] = diag_value;
    }
}

void csr_sparse_matvec_mult_local(const CSR *A_local, const double *x_full, double *y_local)
{
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

int csr_distribute(CSR *A_full, CSR *A_local, int rank, int size, MPI_Comm comm)
{
    int n;

    // Broadcast global matrix size
    if (rank == 0)
    {
        n = A_full->n;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, comm);
    A_local->n = n;

    // Compute row distribution for All ranks
    int base = n / size;
    int remainder = n % size;
    A_local->n_local = (rank < remainder) ? base + 1 : base;
    A_local->row_start = rank * base + (rank < remainder ? rank : remainder); // starting row index in global matrix

    // Step 1: Scatter row_ptr segments
    int *local_row_ptr_temp = (int *)malloc((A_local->n_local + 1) * sizeof(int));

    if (!local_row_ptr_temp)
    {
        fprintf(stderr, "Error allocating memory for local_row_ptr_temp\n");
        return -1;
    }

    if (rank == 0)
    {
        int *sendcounts = (int *)malloc(size * sizeof(int));
        int *displs = (int *)malloc(size * sizeof(int));

        for (int i = 0; i < size; i++)
        {
            int start = i * base + (i < remainder ? i : remainder);
            int count = ((i < remainder) ? (base + 1) : base);
            sendcounts[i] = count + 1; // +1 for row_ptr
            displs[i] = start;
        }

        MPI_Scatterv(A_full->row_ptr, sendcounts, displs, MPI_INT,
                     local_row_ptr_temp, A_local->n_local + 1, MPI_INT,
                     0, comm);

        free(sendcounts);
        free(displs);
    }
    else
    {
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT,
                     local_row_ptr_temp, A_local->n_local + 1, MPI_INT,
                     0, comm);
    }

    // Adjust local_row_ptr to start from 0
    int nnz_offset = local_row_ptr_temp[0];
    A_local->nnz = local_row_ptr_temp[A_local->n_local] - nnz_offset;

    // allocate row_ptr for A_local
    A_local->row_ptr = (int *)malloc((A_local->n_local + 1) * sizeof(int));

    if (!A_local->row_ptr)
    {
        fprintf(stderr, "Error allocating memory for local row_ptr\n");
        free(local_row_ptr_temp);
        return -1;
    }

    // Adjust row_ptr to start from 0
    for (int i = 0; i < A_local->n_local + 1; i++)
    {
        A_local->row_ptr[i] = local_row_ptr_temp[i] - nnz_offset;
    }

    free(local_row_ptr_temp);
    // Debug: print allocation size
    if (rank == 0)
    {
        printf("Rank 0: nnz=%d, allocating col_indices (%d ints), values (%d doubles)\n",
               A_local->nnz, A_local->nnz, A_local->nnz);
        double total_size = A_local->nnz * (sizeof(int) + sizeof(double));
        printf("Rank 0: Total allocation size for col_indices and values: %.2f MB\n",
               total_size / (1024.0 * 1024.0));
    }

    // Step 2: Scatter col_indices and values based on each rank's nnz
    // allocate col_indices and values for A_local
    A_local->col_indices = (int *)malloc(A_local->nnz * sizeof(int));

    if (!A_local->col_indices)
    {
        fprintf(stderr, "Rank %d: malloc failed for col_indices (need %d ints = %.2f MB)\n",
                rank, A_local->nnz, A_local->nnz * sizeof(int) / (1024.0 * 1024.0));
        free(A_local->row_ptr);
        return -1;
    }

    A_local->values = (double *)malloc(A_local->nnz * sizeof(double));

    if (!A_local->values)
    {
        fprintf(stderr, "Rank %d: malloc failed for values (need %d doubles = %.2f MB)\n",
                rank, A_local->nnz, A_local->nnz * sizeof(double) / (1024.0 * 1024.0));
        free(A_local->row_ptr);
        free(A_local->col_indices);
        return -1;
    }

    if (rank == 0)
    {
        // Prepare sendcounts and displs for col_indices and values
        int *sendcounts_nnz = (int *)malloc(size * sizeof(int));
        int *displs_nnz = (int *)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++)
        {
            // let say remainder = 2, base = 3, size = 5
            // p0_start = 0 * 3 + 0 = 0,    count = 4
            // p1_start = 1 * 3 + 1 = 4,    count = 4
            // p2_start = 2 * 3 + 2 = 8,    count = 3
            // p3_start = 3 * 3 + 2 = 11,   count = 3
            // p4_start = 4 * 3 + 2 = 14,   count = 3
            int start = i * base + (i < remainder ? i : remainder);
            int count = ((i < remainder) ? (base + 1) : base);
            sendcounts_nnz[i] = A_full->row_ptr[start + count] - A_full->row_ptr[start]; // start to end nnz for this rank
            displs_nnz[i] = A_full->row_ptr[start];
        }

        MPI_Scatterv(A_full->col_indices, sendcounts_nnz, displs_nnz, MPI_INT,
                     A_local->col_indices, A_local->nnz, MPI_INT,
                     0, comm);

        MPI_Scatterv(A_full->values, sendcounts_nnz, displs_nnz, MPI_DOUBLE,
                     A_local->values, A_local->nnz, MPI_DOUBLE,
                     0, comm);

        free(sendcounts_nnz);
        free(displs_nnz);
    }
    else
    {
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT,
                     A_local->col_indices, A_local->nnz, MPI_INT,
                     0, comm);

        MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE,
                     A_local->values, A_local->nnz, MPI_DOUBLE,
                     0, comm);
    }

    return 0;
}

int csr_build_spd_full(CSR *A, int grid_size)
{
    int nx = grid_size;
    int ny = grid_size;
    int nz = grid_size;
    int n = nx * ny * nz;

    A->n = n;
    A->n_local = n; // full matrix on rank 0
    A->row_start = 0;

    // Allocate row_ptr
    A->row_ptr = (int *)malloc((n + 1) * sizeof(int));
    if (!A->row_ptr)
    {
        fprintf(stderr, "Error allocating memory for row_ptr\n");
        return -1;
    }

    int nnz = 0;
    for (int iz = 0; iz < nz; iz++)
    {
        for (int iy = 0; iy < ny; iy++)
        {
            for (int ix = 0; ix < nx; ix++)
            {
                int row = iz * nx * ny + iy * nx + ix;
                A->row_ptr[row] = nnz;

                int count = 0;
                // Loop over 27-point stencil
                for (int sz = -1; sz <= 1; sz++)
                {
                    int iz_n = iz + sz;
                    if (iz_n >= 0 && iz_n < nz)
                    {
                        for (int sy = -1; sy <= 1; sy++)
                        {
                            int iy_n = iy + sy;
                            if (iy_n >= 0 && iy_n < ny)
                            {
                                for (int sx = -1; sx <= 1; sx++)
                                {
                                    int ix_n = ix + sx;
                                    if (ix_n >= 0 && ix_n < nx)
                                    {
                                        count++; // Valid neighbor
                                    }
                                }
                            }
                        }
                    }
                }
                nnz += count;
            }
        }
    }
    A->row_ptr[n] = nnz;
    A->nnz = nnz;

    // Allocate col_indices and values
    A->col_indices = (int *)malloc(nnz * sizeof(int));
    A->values = (double *)malloc(nnz * sizeof(double));
    if (!A->col_indices || !A->values)
    {
        fprintf(stderr, "Failed to allocate CSR arrays\n");
        free(A->row_ptr);
        return -1;
    }

    // Second pass: fill entries (diagonal = 50.0, off-diagonal = -1.0)
    for (int iz = 0; iz < nz; iz++)
    {
        for (int iy = 0; iy < ny; iy++)
        {
            for (int ix = 0; ix < nx; ix++)
            {
                int row = iz * nx * ny + iy * nx + ix;
                int pos = A->row_ptr[row];

                // Loop over 27-point stencil
                for (int sz = -1; sz <= 1; sz++)
                {
                    int iz_n = iz + sz;
                    if (iz_n >= 0 && iz_n < nz)
                    {
                        for (int sy = -1; sy <= 1; sy++)
                        {
                            int iy_n = iy + sy;
                            if (iy_n >= 0 && iy_n < ny)
                            {
                                for (int sx = -1; sx <= 1; sx++)
                                {
                                    int ix_n = ix + sx;
                                    if (ix_n >= 0 && ix_n < nx)
                                    {
                                        int col = iz_n * nx * ny + iy_n * nx + ix_n;

                                        A->col_indices[pos] = col;
                                        if (col == row)
                                        {
                                            A->values[pos] = 50.0; // Diagonal
                                        }
                                        else
                                        {
                                            A->values[pos] = -1.0; // Off-diagonal
                                        }
                                        pos++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // print the matrix memory usage
    double a_value_size = nnz * sizeof(double);
    double a_colind_size = nnz * sizeof(int);
    double a_rowptr_size = (n + 1) * sizeof(int);
    double total_size = a_value_size + a_colind_size + a_rowptr_size;
    printf("CSR Matrix memory usage: \nvalues =  %.2f MB\n        col_indices = %.2f MB\n        row_ptr = %.2f MB\n        total = %.2f MB\n",
           a_value_size / (1024.0 * 1024.0),
           a_colind_size / (1024.0 * 1024.0),
           a_rowptr_size / (1024.0 * 1024.0),
           total_size / (1024.0 * 1024.0));

    return 0;
}

void csr_free(CSR *A)
{
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