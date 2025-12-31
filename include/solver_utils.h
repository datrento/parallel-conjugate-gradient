#pragma once
#include "csr.h"

void csr_local_Ap_contribution(
    const CSR *A_local, const double *p_local, double *Ap_local,
    int n_local, int row_start, int row_end);

void csr_global_Ap_contribution(
    const CSR *A_local, const double *p, double *Ap_local,
    int n_local, int row_start, int row_end);

void check_the_ap_computation_is_valid(
    const CSR *A_local, double *Ap_local, double *p, int n_local, int rank);

void verify_solution(CSR *A_local, double *x, double *b_local, double *Ax_local, int n_local, int rank, MPI_Comm comm);
