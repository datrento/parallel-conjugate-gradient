#pragma once
/**
 * Validates the provided grid size for csr matrix generation.
 * The grid size must be positive and such that n = grid_size^3 fits in an int.
 * Parameters:
 *    grid_size : Size of the grid (grid_size × grid_size × grid_size)
 *    rank      : MPI rank of the calling process (for error reporting)
 * Returns:
 *    0 if valid, -1 if invalid
 */
int validate_grid_size(int grid_size, int rank);