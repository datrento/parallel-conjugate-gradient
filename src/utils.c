#include <stdlib.h>
#include "utils.h"
#include <stdio.h>
#include <stddef.h>
#include <limits.h>
#include <math.h>

int validate_grid_size(int grid_size, int rank)
{
    /***
     * Validates the provided grid size for csr matrix generation.
     * The grid size must be positive and such that n = grid_size^3 fits in an int.
     *
     * Parameters:
     *    grid_size : Size of the grid (grid_size × grid_size × grid_size)
     *    rank      : MPI rank of the calling process (for error reporting)
     *
     * Returns:
     *    0 if valid, -1 if invalid
     ***/
    const int max_grid = (int)cbrt((double)INT_MAX) - 1;

    if (grid_size <= 0)
    {
        if (rank == 0)
        {
            fprintf(stderr, "[rank %d]Error: grid_size must be positive (got %d)\n", rank, grid_size);
            fflush(stderr);
        }
        return -1;
    }

    if (grid_size > max_grid)
    {
        if (rank == 0)
        {
            fprintf(stderr, "[rank %d]Error: grid_size=%d exceeds maximum=%d\n", rank,
                    grid_size, max_grid);
            fprintf(stderr, "  n=grid_size^3 must fit in int (INT_MAX=%d)\n", INT_MAX);
            fflush(stderr);
        }
        return -1;
    }

    return 0;
}