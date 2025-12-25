#include <stdlib.h>
#include "utils.h"
#include <stdio.h>

void generate_symmetric_positive_definite_dense_matrix(double *A, int grid_size)
{
    /***
     * Generates a symmetric positive definite matrix A in dense format.
     * Uses the same 27-point stencil as HPCG but stores in row-major dense format.
     *
     * Parameters:
     *    A         : Output dense matrix (size: n × n, stored row-major)
     *    grid_size : Size of the grid (grid_size × grid_size × grid_size)
     *
     * Returns:
     *    None (A is updated in place)
     ***/
    int nx = grid_size;
    int ny = grid_size;
    int nz = grid_size;
    int n = nx * ny * nz;

    // Initialize matrix to zero
    for (int i = 0; i < n * n; i++)
    {
        A[i] = 0.0;
    }

    // Fill matrix using 27-point stencil
    for (int iz = 0; iz < nz; iz++)
    {
        for (int iy = 0; iy < ny; iy++)
        {
            for (int ix = 0; ix < nx; ix++)
            {
                int row = iz * nx * ny + iy * nx + ix;

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

                                        // A[row][col] in row-major: A[row * n + col]
                                        if (col == row)
                                        {
                                            A[row * n + col] = 50.0; // Diagonal
                                        }
                                        else
                                        {
                                            A[row * n + col] = -1.0; // Off-diagonal
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void get_corresponding_b_and_x0(double *A, double *b, double *x0, int n)
{
    /***
     * Given a matrix A, this function generates a corresponding vector b
     * for a known solution x_known (which is a vector of ones) and initializes x0 to a zero vector.
     *
     * Parameters:
     *    A : Input matrix A in dense format (size: n × n, stored row-major)
     *    b      : Output vector b (size: n)
     *    x0     : Initial guess vector x0 (size: n)
     *    n      : Size of the matrix and vectors
     *
     * Returns:
     *    None (b and x0 are updated in place)
     ***/

    // known solution vector temporarily
    double *x_known = (double *)malloc(n * sizeof(double));

    // known solution x
    for (int i = 0; i < n; i++)
    {
        x_known[i] = 1.0; // for simplicity, let's assume the known solution is a vector of ones
        x0[i] = 0.0;      // initial guess x0 as zero vector
    }
    // initialize b
    for (int i = 0; i < n; i++)
    {
        b[i] = 0.0;

        for (int j = 0; j < n; j++)
        {
            b[i] += A[i * n + j] * x_known[j];
        }
    }

    free(x_known);
}

void export_dense_to_bin(const char *filename, double *A, int n)
{
    /***
     * Exports a dense matrix A to a Matrix Market (.mtx) file.
     ***/
    FILE *f = fopen(filename, "wb");
    if (!f)
    {
        fprintf(stderr, "Error opening %s\n", filename);
        return;
    }

    fwrite(&n, sizeof(int), 1, f);               // number of rows
    fwrite(&n, sizeof(int), 1, f);               // number of columns
    fwrite(A, sizeof(double), (size_t)n * n, f); // row major data
    fclose(f);
}