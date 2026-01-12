#pragma once
#include <mpi.h>
#include "csr.h"

/**
 * Structure to hold performance metrics for the CG solvers.
 * Members:
 *    total_time: total solver-call time (Walltime from barrier to barrier)
 *   iter_time: iteration loop time only
 *   iters: iterations performed
 */
typedef struct
{
    double total_time; // total solver-call time (barrier-to-barrier)
    double iter_time;  // iteration loop time only
    int iters;         // iterations performed
    double comm_time;  // total communication time within iterations
} cg_metrics_t;

/**
 * Solves the linear system Ax = b using the Jacobi-preconditioned Conjugate Gradient method.
 * Parameters:
 *      A: global CSR matrix
 *      G: grid information for domain decomposition
 *      b: global right-hand side vector
 *      x: global solution vector (initial guess on input, solution on output)
 *      max_iter: maximum number of iterations
 *      tol: convergence tolerance
 *      metrics: pointer to cg_metrics_t structure to store performance metrics
 * Returns: void
 **/
void jacobi_preconditioned_cg(
    CSR *A, Grid3D *G,
    double *b, double *x,
    int max_iter, double tol,
    cg_metrics_t *metrics);

/**
 * Solves the linear system Ax = b using the Jacobi-preconditioned Pipelined Conjugate Gradient method.
 * Parameters:
 *      A: global CSR matrix
 *      G: grid information for domain decomposition
 *      b: global right-hand side vector
 *      x: global solution vector (initial guess on input, solution on output)
 *      max_iter: maximum number of iterations
 *      tol: convergence tolerance
 *      metrics: pointer to cg_metrics_t structure to store performance metrics
 * Returns: void
 **/
void jacobi_preconditioned_pipelined_cg(
    CSR *A, Grid3D *G,
    double *b, double *x,
    int max_iter, double tol,
    cg_metrics_t *metrics);
