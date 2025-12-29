#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <limits.h>
#include <mpi.h>
#include "csr.h"
#include "utils.h"
// TODO: profiling and performance analysis
// TODO: add omp parallelization within each MPI process for local computations
// TODO: profile and performance analysis again
// TODO: start writing the report

void verify_solution(CSR *A_local, double *x, double *b_local, double *Ax_local, int n_local, int rank, MPI_Comm comm);
void jacobi_preconditioner_z(double *diag_A, double *r, double *z, int n, int rank);
void jacobi_preconditioned_conjugate_gradient(
    const CSR *A_local,
    double *diag_A_local, double *b_local, double *x0,
    double *x_local, double *r_local, double *p, double *p_local,
    double *z_local, double *Ap_local, int n_local, int n,
    int *recvcounts, int *displs,
    int max_iter, double tol, int rank, MPI_Comm comm);

int main(int argc, char *argv[])
{
    // MPI
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm comm = MPI_COMM_WORLD;
    // Problem size
    int grid_size = 200; // default grid size

    if (argc > 1)
    {
        grid_size = atoi(argv[1]);
    }

    // Validate grid size to prevent integer overflow
    if (validate_grid_size(grid_size, rank) != 0) // part of utils.c
    {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    double initial_guess = 0.0;
    int n = grid_size * grid_size * grid_size;

    // this never be triggered due to prior validation, but just in case
    if (n <= 0)
    {
        if (rank == 0)
        {
            fprintf(stderr, "[rank %d] Error: Computed matrix size n=%d is not positive. Check grid_size=%d for overflow.\n", rank, n, grid_size);
            fflush(stderr);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int base = n / size;
    int remainder = n % size;

    // flag to export csr matrix from rank 0 for verification of SPD properties
    int export_csr = 0;

    // Optimization parameters for Conjugate Gradient
    int max_iter = 1000;
    if (argc > 2)
        max_iter = atoi(argv[2]);
    double tol = 1e-10; // relative tolerance
    if (argc > 3)
        tol = atof(argv[3]);

    // Determine local number of rows and starting row for each process
    int n_local = (rank < remainder) ? base + 1 : base;
    int row_start = rank * base + (rank < remainder ? rank : remainder);

    // Pointers for glocal matrix and vectors
    double *x0 = NULL;
    double *p = NULL;

    // Allocate local vectors and matrix in each process
    double *diag_A_local = (double *)malloc(n_local * sizeof(double));
    double *b_local = (double *)malloc(n_local * sizeof(double));
    double *x_local = (double *)malloc(n_local * sizeof(double));
    double *p_local = (double *)malloc(n_local * sizeof(double));
    double *z_local = (double *)malloc(n_local * sizeof(double));
    double *Ap_local = (double *)malloc(n_local * sizeof(double));
    double *r_local = (double *)malloc(n_local * sizeof(double));

    if (!diag_A_local || !b_local || !x_local || !p_local || !z_local || !Ap_local || !r_local)
    {
        fprintf(stderr, "[rank %d] Error allocating local vectors(diag_A_local, b_local, x_local, p_local, z_local, Ap_local, r_local) of size %d\n", rank, n_local);
        fflush(stderr);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    // Local CSR matrix in each process
    CSR A_local;

    if (rank == 0)
    {
        const char *mtx_filename = "../data/matrix_csr.mtx";

        printf("[rank %d]Building CSR matrix (7-point stencil) on rank 0...\n", rank);
        printf("[rank %d]Matrix size: %d x %d\n", rank, n, n);
        printf("[rank %d]Number of processes: %d\n", rank, size);
        printf("[rank %d]Each process local rows: %d\n", rank, n_local);
        printf("[rank %d] Grid size: %d x %d x %d\n", rank, grid_size, grid_size, grid_size);
        fflush(stdout);

        CSR A_full;

        if (csr_build_spd_full(&A_full, grid_size, rank) != 0)
        {
            fprintf(stderr, "[rank 0] Building full CSR matrix on rank 0\n");
            fflush(stderr);
            MPI_Abort(comm, EXIT_FAILURE);
        }

        // check SPD properties of the full matrix on rank 0
        // if (export_csr > 0)
        verify_spd_properties(&A_full, rank);

        // Debug: print memory usage of full CSR matrix on rank 0
        print_csr_memory_usage(rank, A_full.nnz, A_full.n, "full");

        // Distribute CSR matrix to all processes
        if (csr_distribute(&A_full, &A_local, n, n_local, row_start, rank, size, comm) != 0)
        {
            fprintf(stderr, "[rank 0] Error distributing CSR matrix from rank 0\n");
            fflush(stderr);
            MPI_Abort(comm, EXIT_FAILURE);
        }

        // export csr matrix to mtx for verification of SPD properties before freeing
        if (export_csr > 0)
            export_csr_to_mtx(mtx_filename, &A_full, n, rank);

        // free the full matrix in CSR format on rank 0
        csr_free(&A_full);
    }
    else
    {
        // Non-root ranks still need their local CSR slice
        if (csr_distribute(NULL, &A_local, n, n_local, row_start, rank, size, comm) != 0)
        {
            fprintf(stderr, "[rank %d] Error distributing CSR matrixd\n", rank);
            fflush(stderr);
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }

    // Allocate global vectors in each process
    p = (double *)malloc(n * sizeof(double)); // global p vector for Allgather

    if (!p)
    {
        fprintf(stderr, "[rank %d] Error allocating global vector p of size %d\n", rank, n);
        fflush(stderr);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    // Allocate x0 in each process (initial guess)
    x0 = (double *)malloc(n * sizeof(double));

    if (!x0)
    {
        fprintf(stderr, "[rank %d] Error allocating global vector x0 of size %d\n", rank, n);
        fflush(stderr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    // Initialize x0 and b on rank 0
    for (int i = 0; i < n; i++)
    {
        x0[i] = initial_guess; // initial guess x0 as zero vector
    }

    // Broadcast x0 to all processes (initial guess) to allocate after freeing A_full on rank 0
    // MPI_Bcast(x0, n, MPI_DOUBLE, 0, comm); // initial guess

    // Build b = A * x_known where x_known is a vector of specific values (e.g., all 2.0)
    double *x_known = (double *)malloc(n * sizeof(double));

    if (!x_known)
    {
        fprintf(stderr, "[rank %d] Error allocating x_known of size %d\n", rank, n);
        fflush(stderr);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    // initialize x_known to all ranks
    for (int i = 0; i < n; i++)
    {
        x_known[i] = 2.0;
    }

    // b = A * x_known using local sparse matvec multiplication each process computes its local part
    // and stores in b_local
    csr_sparse_matvec_mult_local(&A_local, x_known, b_local);
    free(x_known);

    // get diagonal elements from local CSR matrix
    csr_get_diagonal_local(&A_local, diag_A_local, rank);

    // initialize local x from x0
    for (int i = 0; i < n_local; i++)
    {
        x_local[i] = x0[row_start + i];
    }

    // Precompute recvcounts and displs for variable-lenghth gatherv
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));

    if (!recvcounts || !displs)
    {
        fprintf(stderr, "[rank %d] Error allocating memory for recvcounts or displs\n", rank);
        fflush(stderr);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    if (rank == 0)
        printf("[rank %d] Precomputing recvcounts and displs for Allgatherv...\n", rank);

    // Rank order preserved while gathering n_local from all ranks
    MPI_Allgather(&n_local, 1, MPI_INT, recvcounts, 1, MPI_INT, comm);

    displs[0] = 0;
    for (int i = 1; i < size; i++)
    {
        displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    // Sanity: each rank's block starts at displs[rank] and has length recvcounts[rank]
    if (displs[rank] != row_start || recvcounts[rank] != n_local)
    {
        fprintf(stderr, "[rank %d]Rank %d: mismatch displs=%d recvcounts=%d vs row_start=%d n_local=%d\n",
                rank, rank, displs[rank], recvcounts[rank], row_start, n_local);
        fflush(stderr);
        MPI_Abort(comm, 1);
    }

    // Timing
    double local_time = 0.0;

    MPI_Barrier(comm); // synchronize before timing

    // Solve Ax = b using Jacobi Preconditioned Conjugate Gradient method
    local_time = MPI_Wtime();
    jacobi_preconditioned_conjugate_gradient(
        &A_local, diag_A_local, b_local, x0,
        x_local, r_local, p, p_local,
        z_local, Ap_local, n_local, n,
        recvcounts, displs,
        max_iter, tol, rank, comm);
    local_time = MPI_Wtime() - local_time;
    if (rank == 0)
    {
        printf("[rank %d] Completed Jacobi Preconditioned Conjugate Gradient solver.\n", rank);
        fflush(stdout);
    }

    //  Gather the local x (solution) to x0 in all processes
    MPI_Allgatherv(x_local, n_local, MPI_DOUBLE, x0, recvcounts, displs, MPI_DOUBLE, comm);

    // the minimum and maximum time taken among all processes
    double min_time = 0.0, max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, comm);

    // Verification the solutions
    verify_solution(&A_local, x0, b_local, Ap_local, n_local, rank, comm);

    if (rank == 0)
    {
        printf("[rank %d]Wall-clock time taken for Jacobi Preconditioned Conjugate Gradient: max %.6f s\n",
               rank, max_time);
        printf("[rank %d]Wall-clock time taken for Jacobi Preconditioned Conjugate Gradient: min %.6f s\n",
               rank, min_time);
        fflush(stdout);

        // store the max_time for performance analysis later
        const char *time_filename = "output/jcgtimes.txt";

        // Check if file exists to decide whether to write header
        int file_exists = 0;
        FILE *check_file = fopen(time_filename, "r");
        if (check_file)
        {
            file_exists = 1;
            fclose(check_file);
        }

        FILE *time_file = fopen(time_filename, "a");
        if (time_file)
        {
            // Write header if file is new
            if (!file_exists)
            {
                fprintf(time_file, "#grid_size num_processes max_time_seconds min_time_seconds\n");
            }

            // Append grid_size, number of processes, max and min time
            fprintf(time_file, "%d %d %.6f %.6f\n", grid_size, size, max_time, min_time);
            fclose(time_file);
            fprintf(stdout, "[rank %d]Appended time data to %s:  grid_size=%d, num_processes=%d, max_time=%.6f, min_time=%.6f\n", rank, time_filename, grid_size, size, max_time, min_time);
            fflush(stdout);
        }
        else
        {
            fprintf(stderr, "[rank %d] Error opening %s for writing\n", rank, time_filename);
            fflush(stderr);
        }
    }

    // Common frees (all ranks)
    free(x0);
    free(r_local);
    free(p_local);
    free(Ap_local);
    free(diag_A_local);
    free(z_local);
    free(x_local);
    free(b_local);
    free(p);
    free(recvcounts);
    free(displs);
    csr_free(&A_local);
    MPI_Finalize();
    return 0;
}

void jacobi_preconditioner_z(double *diag_A, double *r, double *z, int n, int rank)
{
    /***
     * Jacobi preconditioner: z = M^-1 * r where M = diag(A) => z = diag(A)^-1 * r
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

void verify_solution(CSR *A_local, double *x, double *b_local, double *Ax_local, int n_local, int rank, MPI_Comm comm)
{
    /***
     * Verifies the solution by computing Ax and comparing it to b.
     * Note: Ax buffer will be overwritten with A*x computation.
     ***/
    csr_sparse_matvec_mult_local(A_local, x, Ax_local);

    double local_err2 = 0.0, local_b2 = 0.0;
    for (int i = 0; i < n_local; i++)
    {
        double d = Ax_local[i] - b_local[i];
        local_err2 += d * d;
        local_b2 += b_local[i] * b_local[i];
    }
    double err2 = 0.0, b2 = 0.0;
    MPI_Reduce(&local_err2, &err2, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_b2, &b2, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    if (rank == 0)
    {
        printf("[rank %d]Verification: ||Ax - b|| = %.6e, ||Ax - b|| / ||b|| (relative) = %.6e\n",
               rank, sqrt(err2), sqrt(err2) / sqrt(b2));
    }
}
void csr_local_Ap_contribution(
    const CSR *A_local,
    const double *p_local,
    double *Ap_local,
    int n_local,
    int row_start,
    int row_end)
{
    for (int i = 0; i < n_local; i++)
    {
        double sum = 0.0;

        for (int j = A_local->row_ptr[i]; j < A_local->row_ptr[i + 1]; j++)
        {
            int col = A_local->col_indices[j];

            // check if col index belongs to local part of p (for each value of A_local row check if the corresponding p is local)
            if (col >= row_start && col < row_end)
            {
                // local part of p
                sum += A_local->values[j] * p_local[col - row_start]; // adjust index for local p
            }
        }
        Ap_local[i] = sum;
    }
}

void csr_global_Ap_contribution(
    const CSR *A_local,
    const double *p,
    double *Ap_local,
    int n_local,
    int row_start,
    int row_end)
{
    for (int i = 0; i < n_local; i++)
    {
        double sum = Ap_local[i];
        for (int j = A_local->row_ptr[i]; j < A_local->row_ptr[i + 1]; j++)
        {
            int col = A_local->col_indices[j];
            // check if col index belongs to remote part of p
            if (col < row_start || col >= row_end)
            {
                // remote part of p
                sum += A_local->values[j] * p[col];
            }
        }
        Ap_local[i] = sum;
    }
}

void check_the_ap_computation_is_valid(const CSR *A_local, double *Ap_local, double *p, int n_local, int rank)
{
    static int checked_once = 0;
    if (!checked_once)
    {
        double *Ap_ref = (double *)malloc(n_local * sizeof(double));
        csr_sparse_matvec_mult_local(A_local, p, Ap_ref);

        double maxdiff = 0.0;

        for (int i = 0; i < n_local; i++)
        {
            double diff = fabs(Ap_local[i] - Ap_ref[i]);
            if (diff > maxdiff)
                maxdiff = diff;
        }

        if (rank == 0)
        {
            printf("[rank %d][DEBUG] Ap split vs max diff: %.6e\n", rank, maxdiff);
            fflush(stdout);
        }

        free(Ap_ref);
        checked_once = 1;
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
     * A_local: local CSR matrix size: (nnz_local for values and col_indices; row_ptr of size n_local+1)
     * diag_A_local: diagonal elements of local CSR matrix
     * b_local: local right-hand side vector size: (n_local)
     * x0: initial guess (global) size: (n)
     * x_local: local solution vector size: (n_local)
     * r_local: local residual vector size: (n_local)
     * p: global search direction vector size: (n)
     * p_local: local search direction vector size: (n_local)
     * z_local: local preconditioned residual vector size: (n_local)
     * Ap_local: local A*p vector size: (n_local)
     * n_local: number of local rows
     * n: global number of rows
     * recvcounts, displs: for Allgatherv of p vector size: (number of processes)
     * max_iter: maximum number of iterations
     * tol: relative tolerance for convergence
     */

    // intermediate scalars
    double alpha = 0.0, beta = 0.0, rsn0 = 0.0, rsnnew = 0.0, rtzold = 0.0, rtznew = 0.0;

    // Ap_local = A_local * x0 use Ap_local as temporary storage for Ax0_local( to resuse the buffer later)
    csr_sparse_matvec_mult_local(A_local, x0, Ap_local);

    // r0 = b - Ax0
    for (int i = 0; i < n_local; i++)
    {
        r_local[i] = b_local[i] - Ap_local[i];
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

    // Status of the solver for convergence
    typedef enum
    {
        CG_NOT_CONVERGED = 0,
        CG_CONVERGED_RESIDUAL = 1,
        CG_BREAKDOWN_PAP = 2,
        CG_BREAKDOWN_RTZ = 3
    } cg_status_t;

    cg_status_t status = CG_NOT_CONVERGED;

    const int row_start = displs[rank];
    const int row_end = row_start + n_local;

    // Main iteration loop
    for (int k = 0; k < max_iter; k++)
    {
        // Gather p_local to p (global)
        // communication bottleneck here
        // MPI_Allgatherv(p_local, n_local, MPI_DOUBLE, p, recvcounts, displs, MPI_DOUBLE, comm);

        // replace the above Allgatherv with non-blocking version
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
}
