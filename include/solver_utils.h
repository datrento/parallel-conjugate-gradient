#pragma once
#include "csr.h"
#include "halo_exchange.h"

void csr_local_Ap_contribution(
    const CSR *A_local, const double *p_local, double *Ap_local,
    int n_local, int row_start, int row_end);

void csr_global_Ap_contribution(
    const CSR *A_local, const double *p, double *Ap_local,
    int n_local, int row_start, int row_end);

void check_the_ap_computation_is_valid(
    const CSR *A_local, double *Ap_local, double *p, int n_local, int rank);

void verify_solution(CSR *A_local, double *x, double *b_local, double *Ax_local, int n_local, int rank, MPI_Comm comm);

void analyze_communication_pattern(
    const CSR *A_local,
    int n_local,
    int row_start,
    int row_end,
    int rank,
    int size,
    int n);

// New function: SpMV using halo exchange
void csr_spmv_halo(
    const CSR *A_local,
    const double *p_local,
    double *Ap_local,
    const HaloExchange *halo,
    int n_local,
    int row_start,
    int row_end);