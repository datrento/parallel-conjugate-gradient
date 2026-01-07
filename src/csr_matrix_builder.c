#include "csr.h"
#include <stdlib.h>
#include <stdio.h>
#include "csr_matrix_builder.h"
#include <limits.h>

void build_local_matrix(CSR *A, Grid3D *G, int rank)
{
    /**
     * Implements a 7-point stencil for a 3D grid and constructs the local CSR matrix.
     * Handles proper partitioning of the grid among MPI processes, including remainders.
     * the diagonal entries are stored and their inverses computed for Jacobi preconditioning.
     * it splits the cubic domain into slabs along the z-dimension.
     * Credits: adopted from hpcg benchmark c++ implementation https://github.com/hpcg-benchmark/hpcg and GitHub Copilot
     **/
    int nx = G->nx, ny = G->ny;
    int n_local = nx * ny * G->local_nz;
    A->n = nx * ny * G->nz;
    A->n_local = n_local;
    A->row_start = G->z_start * nx * ny;

    A->row_ptr = (int *)malloc((n_local + 1) * sizeof(int));
    A->inv_diag = (double *)malloc((n_local > 0 ? n_local : 1) * sizeof(double));

    if (!A->row_ptr || !A->inv_diag)
    {
        fprintf(stderr, "[rank %d] Error allocating CSR row_ptr or inv_diag\n", rank);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int nnz_count = 0;
    for (int lz = 0; lz < G->local_nz; lz++)
    {
        // LOOP through each plane of nx * ny and count non-zeros (count the number of non-zeros per row)
        for (int iy = 0; iy < ny; iy++)
        {
            for (int ix = 0; ix < nx; ix++)
            {
                // count the number of non-zeros for this row: can you give me the logic here ? each row corresponds to a
                // grid point (ix, iy, lz) and contains connections to its 6 neighbors + itself and toal values in each row is between 1 and 7
                // size of the row nx * ny * local_nz
                int gz = lz + G->z_start; // each z level in global coords (plane index)
                int count = 1;            // Diag
                if (gz > 0)               // if it's above there would be a bottom neighbor
                    count++;
                if (gz < G->nz - 1) // if it's below there would be a top neighbor
                    count++;
                if (iy > 0) // if it's not on the left edge there would be a left neighbor
                    count++;
                if (iy < ny - 1) // if it's not on the right edge there would be a right neighbor
                    count++;
                if (ix > 0) // if it's not on the front edge there would be a front neighbor
                    count++;
                if (ix < nx - 1) // if it's not on the back edge there would be a back neighbor
                    count++;
                nnz_count += count;
            }
        }
    }

    A->nnz = nnz_count;

    A->col_indices = (int *)malloc(nnz_count * sizeof(int));
    A->values = (double *)malloc(nnz_count * sizeof(double));

    if (!A->col_indices || !A->values)
    {
        fprintf(stderr, "[rank %d] Error allocating CSR col_indices or values\n", rank);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int current_pos = 0;
    int plane_size = nx * ny;
    for (int lz = 0; lz < G->local_nz; lz++)
    {
        int gz = lz + G->z_start; // global z index
        for (int iy = 0; iy < ny; iy++)
        {
            for (int ix = 0; ix < nx; ix++)
            {
                int row_idx = lz * plane_size + iy * nx + ix; // local row index for ix=0, iy=0, lz=0 it's the first row of plane lz=0 => row_idx = 0
                                                              // for ix=1, iy=0, lz=0 => row_idx =1
                                                              // for ix=2, iy=0, lz=0 => row_idx =2
                                                              // for ix=0, iy=1, lz=0 => row_idx =nx
                A->row_ptr[row_idx] = current_pos;            // start of this row in col_indices and values arrays
                int global_row = row_idx + A->row_start;      // global row index in the full matrix
                int neighbors = 0;
                int diag_idx = -1;

                // for each row we will fill in the column indices and values based on the 7-point stencil
                int offsets[7][3] = {{0, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}}; // offsets for the 7-point stencil
                for (int s = 0; s < 7; s++)                                                                           // maximum of 7 non zeros per row
                {
                    int nz_g = gz + offsets[s][0], ny_g = iy + offsets[s][1], nx_g = ix + offsets[s][2]; // global coords of neighbor
                    // for first row of z=0, iy=0, ix=0 nz_g = 0, ny_g=0, nx_g=0 => itself
                    // for first row of z=0, iy=0, ix=1 nz_g = 0, ny_g=0, nx_g=1 => right neighbor
                    // for first row of z=0, iy=0, ix=2 nz_g = 0, ny_g=0, nx_g=2 => right neighbor
                    if (nz_g >= 0 && nz_g < G->nz && ny_g >= 0 && ny_g < ny && nx_g >= 0 && nx_g < nx)
                    {
                        int g_col = nz_g * plane_size + ny_g * nx + nx_g;
                        A->col_indices[current_pos] = g_col;
                        if (g_col == global_row)    // diagonal entry
                            diag_idx = current_pos; // store position of diagonal entry
                        else
                        {
                            A->values[current_pos] = -1.0; // off-diagonal entries are -1
                            neighbors++;
                        }
                        current_pos++;
                    }
                }
                double d_val = (double)neighbors + 0.1; // diagonal value is number of neighbors + 0.1 (to ensure it's diagonally dominant)
                A->values[diag_idx] = d_val;            // set diagonal entry
                A->inv_diag[row_idx] = 1.0 / d_val;
            }
        }
    }
    A->row_ptr[n_local] = current_pos; // end pointer for the last row

    if (rank == 0)
    {
        printf("[rank %d] Local CSR matrix built: n_local=%d, nnz=%d\n", rank, A->n_local, A->nnz);
        fflush(stdout);
    }
}