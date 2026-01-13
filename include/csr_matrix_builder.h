#pragma once
#include "csr.h"
/**
 * Builds the local CSR matrix for the 3D Poisson problem using a 7-point stencil.
 * Parameters:
 *      A: pointer to CSR matrix to be filled
 *      G: pointer to Grid3D structure with grid and decomposition info
 *      rank: MPI rank for logging purposes
 * Returns: void
 */
void build_local_matrix(CSR *A, Grid3D *G, int rank);

/**
 * Builds the local CSR matrix for the 3D Poisson problem using a 27-point stencil.
 * Parameters:
 *      A: pointer to CSR matrix to be filled
 *      G: pointer to Grid3D structure with grid and decomposition info
 *      rank: MPI rank for logging purposes
 * Returns: void
 */


void build_local_matrix_27stencil(CSR *A, Grid3D *G, int rank);