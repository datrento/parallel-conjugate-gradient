#pragma once
#include "csr.h"
// Rank 0 builds the full SPD matrix of size grid_size^3 x grid_size^3 in CSR format
int csr_build_spd_full(CSR *A, int grid_size, int rank);