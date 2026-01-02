#include "solver.h"
#include "config.h"
#include "solver_utils.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "halo_exchange.h"

static void jacobi_preconditioner_z(double *diag_A, double *r, double *z, int n, int rank)
{
    /***
     * Jacobi preconditioner: z = M^-1 * r where M = diag(A) => z = diag(A)^-1 * r
     * Parameters:
     *     diag_A: diagonal elements of A (size n)
     *     r: residual vector (size n)
     *     z: preconditioned vector (size n)
     *     n: size of vectors
     *     rank: MPI rank for logging purposes
     * Returns: void
     ***/
    for (int i = 0; i < n; i++)
    {
        if (diag_A[i] == 0.0)
        {
            fprintf(stderr, "[rank %d]Error: Diagonal element diag_A[%d] is zero.\n", rank, i);
            fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); // though it's check before when we extract diagonal elements per rank
        }

        z[i] = r[i] / diag_A[i];
    }
}

void jacobi_preconditioned_conjugate_gradient(
    const CSR *A_local,
    double *diag_A_local, double *b_local, double *x0,
    double *x_local, double *r_local, double *p, double *p_local,
    double *z_local, double *Ap_local, int n_local, int n,
    int *recvcounts, int *displs,
    int max_iter, double tol, int rank, MPI_Comm comm)
{
    /**
     * Jacobi Preconditioned Conjugate Gradient Method to solve Ax = b
     * Parameters:
     *      A_local: local CSR matrix size: (nnz_local for values and col_indices; row_ptr of size n_local+1)
     *      diag_A_local: diagonal elements of local CSR matrix
     *      b_local: local right-hand side vector size: (n_local)
     *      x0: initial guess (global) size: (n)
     *      x_local: local solution vector size: (n_local)
     *      r_local: local residual vector size: (n_local)
     *      p: global search direction vector size: (n)
     *      p_local: local search direction vector size: (n_local)
     *      z_local: local preconditioned residual vector size: (n_local)
     *      Ap_local: local A*p vector size: (n_local)
     *      n_local: number of local rows
     *      n: global number of rows
     *      recvcounts, displs: for Allgatherv of p vector size: (number of processes)
     *      max_iter: maximum number of iterations
     *      tol: relative tolerance for convergence
     *      rank: MPI rank
     *      comm: MPI communicator
     * Returns: void (solution is in x_local)
     */

    // intermediate scalars
    double alpha = 0.0, beta = 0.0, rsn0 = 0.0, rsnnew = 0.0, rtzold = 0.0, rtznew = 0.0;

    // Status of the solver for convergence
    typedef enum
    {
        CG_NOT_CONVERGED = 0,
        CG_CONVERGED_RESIDUAL = 1,
        CG_BREAKDOWN_PAP = 2,
        CG_BREAKDOWN_RTZ = 3
    } cg_status_t;

    cg_status_t status = CG_NOT_CONVERGED;

    // --------------------------------------------------------

#if PCG_ENABLE_OVERLAP || PCG_USE_GHOST_EXCHANGE

    const int row_start = displs[rank];
    const int row_end = row_start + n_local;
#endif

#if PCG_USE_GHOST_EXCHANGE
    // Build halo exchange structure (before CG loop)
    HaloExchange halo;
    int size;
    MPI_Comm_size(comm, &size);

    int halo_status = halo_exchange_build(&halo, A_local, n_local, row_start, row_end,
                                          rank, size, n, comm);
    if (halo_status != 0)
    {
        fprintf(stderr, "[rank %d] Failed to build halo exchange\n", rank);
        return;
    }
#endif
    // --------------------------------------------------------

    if (x0) // non-zero initial guess in case x0 is provided
    {
        // Ap_local = A_local * x0 use Ap_local as temporary storage for Ax0_local( to resuse the buffer later)
        csr_sparse_matvec_mult_local(A_local, x0, Ap_local);

        // r0 = b - Ax0
        for (int i = 0; i < n_local; i++)
        {
            r_local[i] = b_local[i] - Ap_local[i];
        }
    }
    else
    {
        // if initial guess is zero, Ap_local = 0
        for (int i = 0; i < n_local; i++)
        {
            Ap_local[i] = 0.0;
        }

        // r0 = b - 0 = b
        for (int i = 0; i < n_local; i++)
        {
            r_local[i] = b_local[i];
        }
    }

    // z0 = M^-1 * r0 (Jacobi preconditioner) M^-1 = diag(A)^-1
    jacobi_preconditioner_z(diag_A_local, r_local, z_local, n_local, rank);

    // p0 = z0
    for (int i = 0; i < n_local; i++)
    {
        p_local[i] = z_local[i];
    }

    // ||r0||^2
    rsn0 = 0.0;
    for (int i = 0; i < n_local; i++)
    {
        rsn0 += r_local[i] * r_local[i];
    }

    // rtz0 = r0^T * z0
    rtzold = 0.0;
    for (int i = 0; i < n_local; i++)
    {
        rtzold += r_local[i] * z_local[i];
    }

    // Global reductions for rsn0 and rtz0 combined
    double rsn0_rtzold_sum[2] = {rsn0, rtzold};
    MPI_Allreduce(MPI_IN_PLACE, rsn0_rtzold_sum, 2, MPI_DOUBLE, MPI_SUM, comm);
    rsn0 = rsn0_rtzold_sum[0];
    rtzold = rsn0_rtzold_sum[1];

    // #if PCG_ENABLE_OVERLAP
    //     row_start = displs[rank];
    //     row_end = row_start + n_local;
    // #endif

    // Main iteration loop
    for (int k = 0; k < max_iter; k++)
    {
// Gather p_local to p (global)
// communication bottleneck here
#if PCG_ENABLE_OVERLAP
        // Overlap communication and computation
        MPI_Request request;
        MPI_Iallgatherv(p_local, n_local, MPI_DOUBLE, p, recvcounts, displs, MPI_DOUBLE, comm, &request);

        // While waiting for p to be gathered, we can perform other computations if needed
        // Compute local-only Ap contribution while p is gathering
        // Ap_local = A_local * p_local
        csr_local_Ap_contribution(A_local, p_local, Ap_local, n_local, row_start, row_end);

        // Wait for full p to arrive
        MPI_Wait(&request, MPI_STATUS_IGNORE);

        // Finish remote contributions using gathered p
        // Ap_local += A_local * p (remote contributions)
        csr_global_Ap_contribution(A_local, p, Ap_local, n_local, row_start, row_end);
#else
#if PCG_USE_GHOST_EXCHANGE
        // NEW CODE:
        // Exchange ghost values
        halo_exchange_execute(&halo, p_local, comm);

        // Compute SpMV using halo values
        csr_spmv_halo(A_local, p_local, Ap_local, &halo, n_local,
                      row_start, row_end);
#else
        // // Standard blocking Allgatherv
        MPI_Allgatherv(p_local, n_local, MPI_DOUBLE, p, recvcounts, displs, MPI_DOUBLE, comm);
        // // A = A * p
        csr_sparse_matvec_mult_local(A_local, p, Ap_local);
#endif

#endif

// check if the Ap_local is valid
#ifdef DEBUG
        check_the_ap_computation_is_valid(A_local, Ap_local, p, n_local, rank);
#endif

        // alpha = rtz / (p^T * Ap)
        double pAp = 0.0;
        for (int i = 0; i < n_local; i++)
        {
            pAp += p_local[i] * Ap_local[i];
        }

        MPI_Allreduce(MPI_IN_PLACE, &pAp, 1, MPI_DOUBLE, MPI_SUM, comm);

        if (fabs(pAp) < 1e-20)
        {
            fprintf(stderr, "[rank %d] Error: Division by zero encountered in iteration %d.\n", rank, k);
            fflush(stderr);
            status = CG_BREAKDOWN_PAP;
            break; // Exit the loop to prevent division by zero
        }

        // global value of alpha
        alpha = rtzold / pAp;

        // xk+1 = xk + alpha * pk
        for (int i = 0; i < n_local; i++)
        {
            x_local[i] += alpha * p_local[i];
        }

        // rk+1 = rk - alpha * Apk
        for (int i = 0; i < n_local; i++)
        {
            r_local[i] -= alpha * Ap_local[i];
        }

        // zk+1 = M^-1 * rk+1 (Jacobi preconditioner)
        jacobi_preconditioner_z(diag_A_local, r_local, z_local, n_local, rank);

        // rsnnew = r^T * r
        rsnnew = 0.0;
        for (int i = 0; i < n_local; i++)
        {
            rsnnew += r_local[i] * r_local[i];
        }

        // rtznew = r^T * z
        rtznew = 0.0;
        for (int i = 0; i < n_local; i++)
        {
            rtznew += r_local[i] * z_local[i];
        }

        // Global reductions for rsnnew and rtznew combined
        double sums[2] = {rsnnew, rtznew};
        MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE, MPI_SUM, comm);
        rsnnew = sums[0];
        rtznew = sums[1];

        // Check for convergence
        // for efficiency, we use the relative convergence criterion without computing square roots
        // ||rk+1|| < tol * ||r0||
        if (rsnnew < (tol * tol) * rsn0)
        {
            if (rank == 0)
            {
                fprintf(stdout, "[rank %d] Converged in %d iterations.\n", rank, k + 1);
            }
            fflush(stdout);
            status = CG_CONVERGED_RESIDUAL;
            break;
        }

        if (fabs(rtzold) < 1e-20)
        {
            fprintf(stderr, "[rank %d] Error: Division by zero encountered in iteration %d.\n", rank, k);
            fflush(stderr);
            status = CG_BREAKDOWN_RTZ;
            break; // Exit the loop to prevent division by zero
        }

        // beta = rtznew / rtzold
        beta = rtznew / rtzold;

        // pk+1 = zk+1 + beta * pk
        for (int i = 0; i < n_local; i++)
        {
            p_local[i] = z_local[i] + beta * p_local[i];
        }

        // Update rtzold for next iteration
        rtzold = rtznew;
    }

    // Check if loop ended due to reaching max_iter
    if (rank == 0)
    {
        if (status == CG_NOT_CONVERGED)
            fprintf(stdout, "[rank %d] Reached maximum iterations (%d) without convergence.\n", rank, max_iter);
        fflush(stdout);
    }

#if PCG_USE_GHOST_EXCHANGE
    // Free halo exchange structure
    halo_exchange_free(&halo);
#endif
}
