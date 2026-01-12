#include "solver.h"
#include "solver_utils.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
__attribute__((unused)) static void log_convergence(int rank, int iter, double res_norm, double r0_norm, const char *alg_name)
{
    // Debug: print norms every 5 iterations
    if (iter % 5 == 0 && rank == 0)
    {
        char log_filename[256];
        snprintf(log_filename, sizeof(log_filename), "output/residual_%s.txt", alg_name);

        FILE *log_file = fopen(log_filename, "a");
        if (log_file)
        {
            if (iter == 0)
                fprintf(log_file, "#iteration residual_norm\n");
            fprintf(log_file, "%d %e\n", iter, res_norm / r0_norm);
            fclose(log_file);
        }
    }
}
static void jacobi_preconditioner_z(double *inv_diag, double *r, double *z, int n)
{
    /***
     * Jacobi preconditioner: z = M^-1 * r where M = diag(A) => z = diag(A)^-1 * r
     * Parameters:
     *     inv_diag: inverse diagonal elements of A (size n)
     *     r: residual vector (size n)
     *     z: preconditioned vector (size n)
     *     n: size of vectors
     * Returns: void
     ***/
    for (int i = 0; i < n; i++)
    {
        z[i] = r[i] * inv_diag[i];
    }
}

void jacobi_preconditioned_cg(
    CSR *A, Grid3D *G,
    double *b, double *x,
    int max_iter, double tol,
    cg_metrics_t *metrics)
{
    /**
     * Jacobi Preconditioned CG (baseline) using Allgatherv for slab-based partitions.
     * Matches the z-slab decomposition; uses inverse diagonal preconditioning.
     */

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_local = A->n_local;
    int n = A->n;

    double *r_local = calloc(n_local, sizeof(double));
    double *z_local = calloc(n_local, sizeof(double));
    double *p_local = calloc(n_local, sizeof(double));
    double *Ap_local = calloc(n_local, sizeof(double));
    double *p = calloc(n, sizeof(double));

    // allocation check
    if (!r_local || !z_local || !p_local || !Ap_local || !p)
    {
        fprintf(stderr, "[rank %d] Error: Memory allocation failed in jacobi_preconditioned_cg, r_local or z_local or p_local or Ap_local or p\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // recvcounts/displs for Allgatherv based on plane partition
    int plane_size = G->nx * G->ny;
    int base_nz = G->nz / size;
    int remainder = G->nz % size;
    int *recvcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    if (!recvcounts || !displs)
    {
        fprintf(stderr, "[rank %d] Error: Memory allocation failed for recvcounts or displs in jacobi_preconditioned_cg\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int i = 0; i < size; i++)
    {
        int local_nz_i = base_nz + (i < remainder ? 1 : 0);
        recvcounts[i] = local_nz_i * plane_size;
        displs[i] = (i == 0) ? 0 : displs[i - 1] + recvcounts[i - 1];
    }

    // synchronize before timing total and core iterations; include pre-iteration work in total_time
    MPI_Barrier(MPI_COMM_WORLD);
    double t_total0 = MPI_Wtime();

    // r0 = b - A*x0; start from r=b and apply preconditioner
    memcpy(r_local, b, n_local * sizeof(double));
    jacobi_preconditioner_z(A->inv_diag, r_local, z_local, n_local);
    memcpy(p_local, z_local, n_local * sizeof(double));

    double rsn0 = 0.0, rtzold = 0.0;
    for (int i = 0; i < n_local; i++)
    {
        rsn0 += r_local[i] * r_local[i];
        rtzold += r_local[i] * z_local[i];
    }

    double pair0[2] = {rsn0, rtzold};
    MPI_Allreduce(MPI_IN_PLACE, pair0, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    rsn0 = pair0[0];
    rtzold = pair0[1];

    double t0 = MPI_Wtime();
    int k;

    double t_comm = 0.0; // total communication time accumulator

    for (k = 0; k < max_iter; k++)
    {
        // 1: block comm; time gather comm time
        double tc0 = MPI_Wtime();
        // gather p
        MPI_Allgatherv(p_local, n_local, MPI_DOUBLE, p, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        t_comm += (MPI_Wtime() - tc0);

        csr_sparse_matvec_mult_local(A, p, Ap_local);

        double pAp = 0.0;
        for (int i = 0; i < n_local; i++)
            pAp += p_local[i] * Ap_local[i];

        // 2: block comm; time reduction comm time
        double tc1 = MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE, &pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        t_comm += (MPI_Wtime() - tc1);

        if (fabs(pAp) < 1e-20)
        {
            if (rank == 0)
                fprintf(stderr, "[rank %d] Breakdown: pAp ~ 0 at iter %d\n", rank, k + 1);
            break;
        }

        double alpha = rtzold / pAp;

        for (int i = 0; i < n_local; i++)
        {
            x[i] += alpha * p_local[i];
            r_local[i] -= alpha * Ap_local[i];
        }

        jacobi_preconditioner_z(A->inv_diag, r_local, z_local, n_local);

        double rsnnew = 0.0, rtznew = 0.0;
        for (int i = 0; i < n_local; i++)
        {
            rsnnew += r_local[i] * r_local[i];
            rtznew += r_local[i] * z_local[i];
        }

        // 3: block comm; time reduction comm time
        double pair[2] = {rsnnew, rtznew};
        double tc2 = MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE, pair, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        t_comm += (MPI_Wtime() - tc2);
        rsnnew = pair[0];
        rtznew = pair[1];
        if (rsnnew < (tol * tol) * rsn0)
        {
            if (rank == 0)
                fprintf(stdout, "[rank %d] Converged at iteration %d with residual %e.\n", rank, k + 1, sqrt(rsnnew));
            break;
        }

        if (fabs(rtzold) < 1e-20)
        {
            if (rank == 0)
                fprintf(stderr, "[rank %d] Breakdown: rtzold ~ 0 at iter %d\n", rank, k + 1);
            break;
        }

#ifdef DEBUG
        log_convergence(rank, k, sqrt(rsnnew), sqrt(rsn0), "baseline");
#endif

        double beta = rtznew / rtzold;

        for (int i = 0; i < n_local; i++)
            p_local[i] = z_local[i] + beta * p_local[i];

        rtzold = rtznew;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    if (metrics)
    {
        metrics->iters = k + 1;
        metrics->iter_time = (t1 - t0);
        metrics->total_time = (t1 - t_total0);
        metrics->comm_time = t_comm;
    }

    goto cleanup;

cleanup:
    free(r_local);
    free(z_local);
    free(p_local);
    free(Ap_local);
    free(p);
    free(recvcounts);
    free(displs);
}

void jacobi_preconditioned_pipelined_cg(
    CSR *A, Grid3D *G,
    double *b, double *x,
    int max_iter, double tol,
    cg_metrics_t *metrics)
{
    /**
     * Pipelined CG with Jacobi preconditioning and halo-aware SpMV.
     * Matches the z-slab domain decomposition used in build_local_matrix.
     */

    /**
     * Implements the pipelined Conjugate Gradient method with Jacobi preconditioning.
     * Uses non-blocking MPI communication to overlap halo exchanges with computation.
     *
     * Parameters:
     * - A_local: Local CSR matrix
     * - G_local: Local grid information for partitioning and neighbor ranks
     * - b_local: Local Right-hand side vector
     * - x_local: Local solution vector (initial guess and output)
     * - max_iter: Maximum number of iterations
     * - tol: Convergence tolerance
     **/

    int n_local = A->n_local;       // number of local rows
    int plane_size = G->nx * G->ny; // size of one xy-plane
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // better to use calloc to initialize to zero
    double *r = malloc((n_local) * sizeof(double)), *u = malloc((n_local) * sizeof(double)),
           *w = malloc((n_local) * sizeof(double)), *m = malloc((n_local) * sizeof(double)),
           *v = malloc((n_local) * sizeof(double)), *p = calloc(n_local, sizeof(double)),
           *s = calloc(n_local, sizeof(double)), *q = calloc(n_local, sizeof(double)),
           *z = calloc(n_local, sizeof(double));
    if (!r || !u || !w || !m || !v || !p || !s || !q || !z)
    {
        fprintf(stderr, "[rank %d] Error: Memory allocation failed in jacobi_preconditioned_pipelined_cg, r, w, m, v, p, s, q, or z\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    // Halo buffers
    double *r_up = malloc(plane_size * sizeof(double));   // halo buffer for upper neighbor
    double *r_down = malloc(plane_size * sizeof(double)); // halo buffer for lower neighbor

    if (!r_up || !r_down)
    {
        fprintf(stderr, "[rank %d] Error: Memory allocation failed for halo buffers r_up or r_down in jacobi_preconditioned_pipelined_cg\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Request h_req[4], dot_req;
    // h_req[0]: receive from rank up
    // h_req[1]: receive from rank down
    // h_req[2]: send to rank up
    // h_req[3]: send to rank down

    double gamma = 0, gamma_old = 0, delta = 0, alpha = 0, beta = 0;
    double r0_norm = 0;

    // synchronize before timing; include pre-iteration (u0, w0) in total_time
    MPI_Barrier(MPI_COMM_WORLD);
    double t_total0 = MPI_Wtime();

    // r0 := b - A*x0 (assuming x0 = current x). Here we start with r=b.
    memcpy(r, b, n_local * sizeof(double));

    // u0 := M^-1 * r0
    jacobi_preconditioner_z(A->inv_diag, r, u, n_local);

    // w0 := A * u0 (with initial halo exchange)
    // Prepare Halo Exchange for u0 while computing internal SpMV
    halo_exchange_init(G, u, r_up, r_down, h_req);

    // SpMV: w0 = A * u0 (internal + boundary)
    // Split SpMV into internal and boundary parts to overlap with communication
    csr_sparse_matvec_mult_internal(A, G, u, w); // internal rows
    MPI_Waitall(4, h_req, MPI_STATUSES_IGNORE);
    // boundary rows
    csr_sparse_matvec_mult_boundary(A, G, u, r_up, r_down, w);

    // m0 = M^-1 * w0 better to save m0 and use v0 directly
    jacobi_preconditioner_z(A->inv_diag, w, m, n_local);

    // v0 = A * m0 (with initial halo exchange)
    // Prepare Halo Exchange for m0 while computing internal SpMV
    halo_exchange_init(G, m, r_up, r_down, h_req);
    // SpMV: v0 = A * m0 (internal + boundary)
    csr_sparse_matvec_mult_internal(A, G, m, v);
    MPI_Waitall(4, h_req, MPI_STATUSES_IGNORE);
    csr_sparse_matvec_mult_boundary(A, G, m, r_up, r_down, v);

    // -----------------------------------------------------------------------------------------------------
    // Precompute initial dot products
    double local_dots[2] = {0, 0}, global_dots[2];
    for (int i = 0; i < n_local; i++)
    {
        local_dots[0] += r[i] * u[i];
        local_dots[1] += w[i] * u[i];
    }
    MPI_Allreduce(local_dots, global_dots, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    gamma = global_dots[0];
    delta = global_dots[1];
    r0_norm = sqrt(gamma);

    // Initial scalars to make the first iteration formula work without 'if'
    beta = 0.0;
    alpha = gamma / delta;
    gamma_old = 1.0; // dummy, won't affect itr=0 because beta is 0

    if (r0_norm < 1e-18)
        goto cleanup;

#ifdef DEBUG
    // --- (Initial Norm for Plotting Purpose) ---
    double local_r0_l2 = 0;
    for (int i = 0; i < n_local; i++)
        local_r0_l2 += r[i] * r[i];
    double global_r0_l2;
    MPI_Allreduce(&local_r0_l2, &global_r0_l2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double r0_l2_norm = sqrt(global_r0_l2);
#endif

    // time the core iteration loop
    double t0 = MPI_Wtime();
    int itr;
    double t_comm = 0.0;

    // Main iteration loop
    for (itr = 0; itr < max_iter; itr++)
    {
        // Update directions with the precomputed beta
        for (int i = 0; i < n_local; i++)
        {
            z[i] = v[i] + beta * z[i];
            q[i] = m[i] + beta * q[i];
            s[i] = w[i] + beta * s[i];
            p[i] = u[i] + beta * p[i];

            x[i] += alpha * p[i];
            r[i] -= alpha * s[i];
            u[i] -= alpha * q[i];
            w[i] -= alpha * z[i];
        }

        // move check convergence here to overlap with the next dot products and SpMV
        if (sqrt(gamma) / r0_norm < tol)
        {
            if (rank == 0)
            {
                printf("[rank %d]Converged at iteration %d with residual %e\n", rank, itr + 1, sqrt(gamma));
            }
            break;
        }

#ifdef DEBUG
        // To make the plot match Baseline, we need the L2 norm, not gamma
        double local_ri_l2 = 0;
        for (int i = 0; i < n_local; i++)
            local_ri_l2 += r[i] * r[i];
        double global_ri_l2;
        // This reduction is ONLY for the convergence plot data
        MPI_Allreduce(&local_ri_l2, &global_ri_l2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        log_convergence(rank, itr, sqrt(global_ri_l2), r0_l2_norm, "pipelined");
#endif

        // gamma := (r_i, u_i), delta := (w_i, u_i)
        double local_dots[2] = {0, 0};

        for (int i = 0; i < n_local; i++)
        {
            local_dots[0] += r[i] * u[i];
            local_dots[1] += w[i] * u[i];
        }

        // Start non-blocking reduction to overlap with m_i and SpMV prep
        MPI_Iallreduce(local_dots, global_dots, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &dot_req);

        // m_i := M^-1 * w_i
        // for (int i = 0; i < n_local; i++)
        //     m[i] = w[i] * A->inv_diag[i];
        jacobi_preconditioner_z(A->inv_diag, w, m, n_local);

        // v_i := A * m_i
        // (Prepare Halo Exchange for m_i while reduction is in flight)
        halo_exchange_init(G, m, r_up, r_down, h_req);

        // SpMV: v_i = A * m_i (internal + boundary)
        csr_sparse_matvec_mult_internal(A, G, m, v); // internal rows

        // 1: block comm; time halo comm time
        double tc0 = MPI_Wtime();
        MPI_Waitall(4, h_req, MPI_STATUSES_IGNORE);
        t_comm += (MPI_Wtime() - tc0);
        csr_sparse_matvec_mult_boundary(A, G, m, r_up, r_down, v);

        // 2: block comm; time reduction comm time
        double tc1 = MPI_Wtime();
        // Ensure dots are finished
        MPI_Wait(&dot_req, MPI_STATUS_IGNORE);
        t_comm += (MPI_Wtime() - tc1);
        gamma_old = gamma;
        gamma = global_dots[0];
        delta = global_dots[1];

        // alpha_i and beta_i
        beta = gamma / gamma_old;
        alpha = gamma / (delta - (beta * gamma / alpha));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    if (metrics)
    {
        metrics->iters = itr + 1; // iterations performed
        metrics->iter_time = (t1 - t0);
        metrics->total_time = (t1 - t_total0);
        metrics->comm_time = t_comm;
    }

    goto cleanup;

cleanup:
    // Free allocated memory
    free(r);
    free(u);
    free(w);
    free(m);
    free(v);
    free(p);
    free(s);
    free(q);
    free(z);
    free(r_up);
    free(r_down);
}