#include "csr.h"
#include <stdlib.h>
#include <stdio.h>
#include "csr_matrix_builder.h"
#include <limits.h>

int csr_build_spd_full(CSR *A, int grid_size, int rank)
{
    /***
     * Builds a symmetric positive definite (SPD) matrix in CSR format
     * using a 7-point stencil for a 3D grid of size grid_size^3.
     *
     * Parameters:
     *    A         :  Pointer to CSR structure to be filled
     *    grid_size :  Size of the 3D grid (grid_size x grid_size x grid_size)
     *
     * Returns:
     *    0 on success, -1 on failure
     *
     * Credits: adopted from hpcg benchmark c++ implementation https://github.com/hpcg-benchmark/hpcg and GitHub Copilot
     ***/
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
        fprintf(stderr, "[rank %d] Error allocating memory for row_ptr\n", rank);
        fprintf(stderr, "[rank %d] n=%d requires %.2f GB for row_ptr\n",
                rank, n, (n + 1) * sizeof(int) / (1024.0 * 1024.0 * 1024.0));
        fflush(stderr);
        return -1;
    }

    // First pass: count non-zeros per row and build row_ptr
    size_t nnz = 0;
    for (int iz = 0; iz < nz; iz++)
    {
        for (int iy = 0; iy < ny; iy++)
        {
            for (int ix = 0; ix < nx; ix++)
            {
                int row = iz * nx * ny + iy * nx + ix;
                A->row_ptr[row] = (int)nnz;

                int count = 0;
                // Loop over 7-point stencil (face neighbors only)
                for (int sz = -1; sz <= 1; sz++)
                {
                    for (int sy = -1; sy <= 1; sy++)
                    {
                        for (int sx = -1; sx <= 1; sx++)
                        {
                            // Only include center and axis-aligned neighbors
                            int num_nonzero = (sz != 0 ? 1 : 0) + (sy != 0 ? 1 : 0) + (sx != 0 ? 1 : 0);
                            if (num_nonzero > 1)
                                continue; // Skip diagonal/edge/corner neighbors

                            int iz_n = iz + sz;
                            int iy_n = iy + sy;
                            int ix_n = ix + sx;

                            if (iz_n >= 0 && iz_n < nz &&
                                iy_n >= 0 && iy_n < ny &&
                                ix_n >= 0 && ix_n < nx)
                            {
                                count++; // Valid neighbor
                            }
                        }
                    }
                }
                nnz += count;
            }
        }
    }
    A->row_ptr[n] = (int)nnz;

    if (nnz > INT_MAX)
    {
        fprintf(stderr, "[rank %d] Error: nnz=%zu exceeds INT_MAX\n", rank, nnz);
        fflush(stderr);
        free(A->row_ptr);
        return -1;
    }

    A->nnz = (int)nnz;

    // Allocate col_indices and values
    A->col_indices = (int *)malloc(A->nnz * sizeof(int));
    A->values = (double *)malloc(A->nnz * sizeof(double));
    if (!A->col_indices || !A->values)
    {
        // log error and free previously allocated memory
        fprintf(stderr, "[rank %d] Error:  Memory allocation failed for col_indices or values\n", rank);
        fprintf(stderr, "[rank %d] nnz=%zu requires %.2f GB for col_indices and %.2f GB for values\n",
                rank, nnz, nnz * sizeof(int) / (1024.0 * 1024.0 * 1024.0), nnz * sizeof(double) / (1024.0 * 1024.0 * 1024.0));

        // number of bytes allocated so far
        fprintf(stderr, "[rank %d] Allocated %.2f GB for row_ptr\n",
                rank, (n + 1) * sizeof(int) / (1024.0 * 1024.0 * 1024.0));
        fflush(stderr);
        free(A->row_ptr);
        return -1;
    }

    // Second pass:  fill entries with strict diagonal dominance
    // diagonal = (number of off-diagonal neighbors) + 0.1, off-diagonal = -1.0
    for (int iz = 0; iz < nz; iz++)
    {
        for (int iy = 0; iy < ny; iy++)
        {
            for (int ix = 0; ix < nx; ix++)
            {
                int row = iz * nx * ny + iy * nx + ix;
                size_t pos = A->row_ptr[row];

                int num_neighbors = 0; // Count actual off-diagonal neighbors
                size_t diag_pos = 0;   // Track position of diagonal entry

                // Loop over 7-point stencil (face neighbors only)
                for (int sz = -1; sz <= 1; sz++)
                {
                    for (int sy = -1; sy <= 1; sy++)
                    {
                        for (int sx = -1; sx <= 1; sx++)
                        {
                            // Only include center and axis-aligned neighbors
                            int num_nonzero = (sz != 0 ? 1 : 0) + (sy != 0 ? 1 : 0) + (sx != 0 ? 1 : 0);
                            if (num_nonzero > 1)
                                continue; // Skip diagonal neighbors

                            int iz_n = iz + sz;
                            int iy_n = iy + sy;
                            int ix_n = ix + sx;

                            if (iz_n >= 0 && iz_n < nz &&
                                iy_n >= 0 && iy_n < ny &&
                                ix_n >= 0 && ix_n < nx)
                            {
                                int col = iz_n * nx * ny + iy_n * nx + ix_n;
                                A->col_indices[pos] = col;

                                if (col == row)
                                {
                                    diag_pos = pos; // Remember where diagonal is
                                }
                                else
                                {
                                    A->values[pos] = -1.0; // Off-diagonal
                                    num_neighbors++;
                                }
                                pos++;
                            }
                        }
                    }
                }

                // Set diagonal = number of off-diagonal neighbors + epsilon for strict dominance
                A->values[diag_pos] = (double)num_neighbors + 0.1;
            }
        }
    }

    return 0;
}