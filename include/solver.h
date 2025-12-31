#pragma once
#include <mpi.h>
#include "csr.h"
void jacobi_preconditioned_conjugate_gradient(
    const CSR *A_local,
    double *diag_A_local, double *b_local, double *x0,
    double *x_local, double *r_local, double *p, double *p_local,
    double *z_local, double *Ap_local, int n_local, int n,
    int *recvcounts, int *displs,
    int max_iter, double tol, int rank, MPI_Comm comm);