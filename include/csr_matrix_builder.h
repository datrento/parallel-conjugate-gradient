#pragma once
#include "csr.h"
/**
 * Builds the local CSR matrix for the 3D Poisson problem using a 7 and 27-point stencil.
 * Parameters:
 *      A: pointer to CSR matrix to be filled
 *      G: pointer to Grid3D structure with grid and decomposition info
 *      rank: MPI rank for logging purposes
 * Returns: void
 *
 * Note: We only used the 27-point stencil in the experiments reported in the paper.
 */
void build_local_matrix(CSR *A, Grid3D *G, int rank);
void build_local_matrix_27stencil(CSR *A, Grid3D *G, int rank);