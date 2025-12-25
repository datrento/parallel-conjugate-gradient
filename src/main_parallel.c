#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils.h"
#include <math.h>
#include <mpi.h>
#include "csr.h"

void jacobi_preconditioner_z(double *diag_A, double *r, double *z, int n);
void jacobi_preconditioned_conjugate_gradient(
    const CSR *A_local,
    double *diag_A_local, double *b_local, double *x0,
    double *x_local, double *r_local, double *p, double *p_local,
    double *z_local, double *Ap_local, int n_local, int n,
    int *recvcounts, int *displs,
    int max_iter, double tol, MPI_Comm comm);
void get_diagonal_elements(double *A, double *diag_A, int n_local, int start_row, int n);
void vec_mat_mult_global(double *A, double *x, double *Ax, int n);
void vec_mat_mult_local(double *A_local, const double *x, double *Ax_local, int n_local, int n);
void verify_solution(double *Ax, double *x, double *b, int n);

int main(int argc, char *argv[])
{
    // MPI
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm comm = MPI_COMM_WORLD;
    // Problem size
    int grid_size = 240; // default grid size
    int n = grid_size * grid_size * grid_size;
    int base = n / size;
    int remainder = n % size;

    // Determine local number of rows and starting row for each process
    int n_local = (rank < remainder) ? base + 1 : base;
    int row_start = rank * base + (rank < remainder ? rank : remainder);

    // Pointers for glocal matrix and vectors
    double *A = NULL;
    double *b = NULL;
    double *x0 = NULL;
    double *p = NULL;
    double *Ap = NULL;

    // Allocate local vectors and matrix
    // double *A_local = (double *)malloc(n_local * n * sizeof(double));
    double *diag_A_local = (double *)malloc(n_local * sizeof(double));
    double *b_local = (double *)malloc(n_local * sizeof(double));
    double *x_local = (double *)malloc(n_local * sizeof(double));
    double *p_local = (double *)malloc(n_local * sizeof(double));
    double *z_local = (double *)malloc(n_local * sizeof(double));
    double *Ap_local = (double *)malloc(n_local * sizeof(double));
    double *r_local = (double *)malloc(n_local * sizeof(double));
    p = (double *)malloc(n * sizeof(double)); // global p vector for Allgather

    // Allocate x0 in all processes
    x0 = (double *)malloc(n * sizeof(double));

    // CSR handle (local slice for each rank)
    CSR A_local;

    // distribution arrays for Scatterv
    // int *sendcounts_A = NULL;
    // int *displs_A = NULL;
    // int *sendcounts_b = NULL;
    // int *displs_b = NULL;

    if (rank == 0)
    {
        // int need_spd_check = 1; // flag to export matrix for verification

        // printf("Grid size: %d\n", grid_size);
        // printf("Matrix size: %d x %d\n", n, n);

        // A = (double *)malloc(n * n * sizeof(double));
        // generate dense A for given n
        // generate_symmetric_positive_definite_dense_matrix(A, grid_size);
        printf("Building CRS matrix (27-point stencil) on rank 0...\n");

        CSR A_full;
        if (csr_build_spd_full(&A_full, grid_size) != 0)
        {
            fprintf(stderr, "Error building CSR matrix on rank 0\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        printf("Full CSR built: (nxn)=%dx%d\n, nnz=%d (avg %.1f nnz/row)\n",
               A_full.n, A_full.n, A_full.nnz, (double)A_full.nnz / A_full.n);

        // export matrix to binary file for verification of SPD property
        // if (need_spd_check)
        // {
        //     // export matrix to mtx file for verification
        //     export_dense_to_bin("../data/matrix.bin", A, n);
        // }

        // distribute CSR matrix to all processes
        if (csr_distribute(&A_full, &A_local, rank, size, comm) != 0)
        {
            fprintf(stderr, "Error distributing CSR matrix from rank 0\n");
            MPI_Abort(comm, EXIT_FAILURE);
        }

        // free the full matrix in CSR format on rank 0
        csr_free(&A_full);

        // allocate b
        // b = (double *)malloc(n * sizeof(double));

        // generate corresponding b for given A for a known solution x and initialize x0 to 0 vector
        // get_corresponding_b_and_x0(A, b, x0, n);

        // scatter rows of A and corresponding b to all processes
        // sendcounts_A = (int *)malloc(size * sizeof(int));
        // displs_A = (int *)malloc(size * sizeof(int));
        // sendcounts_b = (int *)malloc(size * sizeof(int));
        // displs_b = (int *)malloc(size * sizeof(int));

        // for (int i = 0; i < size; i++)
        // {
        //     sendcounts_A[i] = ((i < remainder) ? (base + 1) : base) * n;        // number of elements to send to each process
        //     displs_A[i] = (i == 0) ? 0 : displs_A[i - 1] + sendcounts_A[i - 1]; // displacement for each process

        //     sendcounts_b[i] = (i < remainder) ? (base + 1) : base;              // number of elements to send to each process
        //     displs_b[i] = (i == 0) ? 0 : displs_b[i - 1] + sendcounts_b[i - 1]; // displacement for each process
        // }

        for (int i = 0; i < n; i++)
        {
            x0[i] = 0.0; // initial guess x0 as zero vector
        }
    }
    else
    {
        // Non-root ranks still need their local CSR slice
        if (csr_distribute(NULL, &A_local, rank, size, comm) != 0)
        {
            fprintf(stderr, "Error distributing CSR matrix to rank %d\n", rank);
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }

    // Broadcast x0 to all processes (initial guess)
    MPI_Bcast(x0, n, MPI_DOUBLE, 0, MPI_COMM_WORLD); // initial guess

    // Scatter A and b to all processes
    // MPI_Scatterv(A, sendcounts_A, displs_A, MPI_DOUBLE, A_local, n_local * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // MPI_Scatterv(b, sendcounts_b, displs_b, MPI_DOUBLE, b_local, n_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Build b = A * x_known where x_known is a vector of ones
    double *x_known = (double *)malloc(n * sizeof(double));
    if (rank == 0)
        for (int i = 0; i < n; i++)
        {
            x_known[i] = 2.0;
        }

    // MPI_Bcast(x_known, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    csr_sparse_matvec_mult_local(&A_local, x_known, b_local);
    free(x_known);

    // get diagonal elements of A per process
    // get_diagonal_elements(A_local, diag_A_local, n_local, row_start, n);
    // get diagonal elements from local CSR matrix
    csr_get_diagonal_local(&A_local, diag_A_local);

    // initialize local x from x0
    for (int i = 0; i < n_local; i++)
    {
        x_local[i] = x0[row_start + i];
    }

    // Precompute recvcounts and displs for variable-lenghth gatherv
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));

    MPI_Allgather(&n_local, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

    displs[0] = 0;
    for (int i = 1; i < size; i++)
    {
        displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    // Sanity: each rank's block starts at displs[rank] and has length recvcounts[rank]
    if (displs[rank] != row_start || recvcounts[rank] != n_local)
    {
        fprintf(stderr, "Rank %d: mismatch displs=%d recvcounts=%d vs row_start=%d n_local=%d\n",
                rank, displs[rank], recvcounts[rank], row_start, n_local);
        MPI_Abort(comm, 1);
    }

    // Parameters for Conjugate Gradient
    int max_iter = 1000;
    double tol = 1e-10;

    // Timing
    double start_time = 0.0, end_time = 0.0;

    start_time = MPI_Wtime();
    jacobi_preconditioned_conjugate_gradient(
        &A_local, diag_A_local, b_local, x0,
        x_local, r_local, p, p_local,
        z_local, Ap_local, n_local, n,
        recvcounts, displs,
        max_iter, tol, comm);
    end_time = MPI_Wtime() - start_time;

    // Gather the local x (solution) to x0 in all processes
    MPI_Allgatherv(x_local, n_local, MPI_DOUBLE, x0, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    double min_time = 0.0, max_time = 0.0;
    MPI_Reduce(&end_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&end_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    // // Verification: compute Ax via CSR SpMV and compare to b
    // csr_sparse_matvec_mult_local(&A_local, x0, Ap_local);

    // // Allocate Ap in root for verification
    // if (rank == 0)
    // {
    //     Ap = (double *)malloc(n * sizeof(double));
    //     b = (double *)malloc(n * sizeof(double));
    // }

    // // All ranks gather Ap_local and b_local to Ap and b in root
    // MPI_Gatherv(Ap_local, n_local, MPI_DOUBLE, Ap, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // MPI_Gatherv(b_local, n_local, MPI_DOUBLE, b, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    csr_sparse_matvec_mult_local(&A_local, x0, Ap_local);

    double local_err2 = 0.0, local_b2 = 0.0;
    for (int i = 0; i < n_local; i++)
    {
        double d = Ap_local[i] - b_local[i];
        local_err2 += d * d;
        local_b2 += b_local[i] * b_local[i];
    }
    double err2 = 0.0, b2 = 0.0;
    MPI_Allreduce(&local_err2, &err2, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&local_b2, &b2, 1, MPI_DOUBLE, MPI_SUM, comm);

    if (rank == 0)
    {
        printf("Verification: ||Ax - b|| = %.6e, relative = %.6e\n",
               sqrt(err2), sqrt(err2) / sqrt(b2));
        printf("Time taken for Jacobi Preconditioned Conjugate Gradient: min %.6f s, max %.6f s\n",
               min_time, max_time);
    }

    // if (rank == 0)
    // {
    //     verify_solution(Ap, x0, b, n);
    //     printf("Time taken for Jacobi Preconditioned Conjugate Gradient: min %.6f s, max %.6f s\n", min_time, max_time);

    //     free(b);
    //     free(Ap);
    // }

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

// void vec_mat_mult_local(const CSR *A_local, const double *x, double *Ax_local, int n_local, int n)
// {
//     /**
//      * Multiplies local dense matrix A_local with vector x to produce local result Ax_local.
//      * Parameters:
//      *    A_local : Local dense matrix A (size: n_local x n, stored row-major)
//      *    x       : Input vector x (size: n)
//      *    Ax_local: Output vector Ax_local (size: n_local)
//      *    n_local : Number of local rows in the matrix
//      *    n       : Total number of columns in the matrix
//      *
//      * Returns:
//      *    None (Ax_local is updated in place)
//      **/
//     for (int i = 0; i < n_local; i++)
//     {
//         Ax_local[i] = 0.0;
//         for (int j = 0; j < n; j++)
//         {
//             Ax_local[i] += A_local[i * n + j] * x[j];
//         }
//     }
// }

// // global matrix-vector multiplication
// void vec_mat_mult_global(double *A, double *x, double *Ax, int n)
// {
//     for (int i = 0; i < n; i++)
//     {
//         Ax[i] = 0.0;
//         for (int j = 0; j < n; j++)
//         {
//             Ax[i] += A[i * n + j] * x[j];
//         }
//     }
// }

// void get_diagonal_elements(double *A_local, double *diag_A_local, int n_local, int start_row, int n)
// {
//     /**
//      * Extracts the diagonal elements from the local dense matrix A_local.
//      * Parameters:
//      *    A_local      : Local dense matrix A (size: n_local x n, stored row-major)
//      *    diag_A_local : Output vector to store diagonal elements (size: n_local)
//      *    n_local      : Number of local rows/columns in the matrix
//      * Returns:
//      *    None (diag_A_local is updated in place)
//      **/

//     for (int i = 0; i < n_local; i++)
//     {
//         int global_row = start_row + i;
//         diag_A_local[i] = A_local[i * n + global_row];

//         if (diag_A_local[i] == 0)
//         {
//             fprintf(stderr, "Error: Zero diagonal element at global row %d.\n", global_row);
//             exit(EXIT_FAILURE);
//         }
//     }
// }
void jacobi_preconditioner_z(double *diag_A, double *r, double *z, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (diag_A[i] == 0.0)
        {
            fprintf(stderr, "Error: Diagonal element diag_A[%d] is zero.\n", i);
            exit(EXIT_FAILURE);
        }

        z[i] = r[i] / diag_A[i];
    }
}

void verify_solution(double *Ax, double *x, double *b, int n)
{
    /***
     * Verifies the solution by computing Ax and comparing it to b.
     * Note: Ax buffer will be overwritten with A*x computation.
     ***/
    double err2 = 0.0, b2 = 0.0;
    for (int i = 0; i < n; i++)
    {
        double d = Ax[i] - b[i];
        err2 += d * d;
        b2 += b[i] * b[i];
    }
    printf("Verification: ||Ax - b|| = %.6e, relative = %.6e\n",
           sqrt(err2), sqrt(err2) / sqrt(b2));

    for (int i = 0; i < 10; i++)
    {
        printf("x[%d] = %.6e\n", i, x[i]);
    }
}

void jacobi_preconditioned_conjugate_gradient(
    const CSR *A_local,
    double *diag_A_local, double *b_local, double *x0,
    double *x_local, double *r_local, double *p, double *p_local,
    double *z_local, double *Ap_local, int n_local, int n,
    int *recvcounts, int *displs,
    int max_iter, double tol, MPI_Comm comm)
{
    double alpha, beta, rsn0, rsnnew, rtzold, rtznew;

    // Ap_local = A_local * x0 use Ap as temporary storage
    csr_sparse_matvec_mult_local(A_local, x0, Ap_local);

    // r0 = b - Ax0
    for (int i = 0; i < n_local; i++)
    {
        r_local[i] = b_local[i] - Ap_local[i];
    }

    // z0 = M^-1 * r0 (Jacobi preconditioner) M^-1 = diag(A)^-1
    jacobi_preconditioner_z(diag_A_local, r_local, z_local, n_local);

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

    MPI_Allreduce(MPI_IN_PLACE, &rsn0, 1, MPI_DOUBLE, MPI_SUM, comm);

    // rtz0 = r0^T * z0
    rtzold = 0.0;
    for (int i = 0; i < n_local; i++)
    {
        rtzold += r_local[i] * z_local[i];
    }

    MPI_Allreduce(MPI_IN_PLACE, &rtzold, 1, MPI_DOUBLE, MPI_SUM, comm);

    // Main iteration loop
    for (int k = 0; k < max_iter; k++)
    {
        MPI_Allgatherv(p_local, n_local, MPI_DOUBLE, p, recvcounts, displs, MPI_DOUBLE, comm);

        // A = A * p
        csr_sparse_matvec_mult_local(A_local, p, Ap_local);

        // alpha = rtz / (p^T * Ap)
        double pAp = 0.0;
        for (int i = 0; i < n_local; i++)
        {
            pAp += p_local[i] * Ap_local[i];
        }

        MPI_Allreduce(MPI_IN_PLACE, &pAp, 1, MPI_DOUBLE, MPI_SUM, comm);

        if (fabs(pAp) < 1e-14)
        {
            fprintf(stderr, "Error: Division by zero encountered in iteration %d.\n", k);
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

        // rsnnew = r^T * r
        rsnnew = 0.0;
        for (int i = 0; i < n_local; i++)
        {
            rsnnew += r_local[i] * r_local[i];
        }

        MPI_Allreduce(MPI_IN_PLACE, &rsnnew, 1, MPI_DOUBLE, MPI_SUM, comm);

        // Check for convergence
        // for efficiency, we use the relative convergence criterion without computing square roots
        // ||rk+1|| < tol * ||r0||
        if (rsnnew < (tol * tol) * rsn0)
        {
            fprintf(stdout, "Converged in %d iterations.\n", k + 1);
            break;
        }

        // zk+1 = M^-1 * rk+1 (Jacobi preconditioner)
        jacobi_preconditioner_z(diag_A_local, r_local, z_local, n_local);

        // rtznew = r^T * z
        rtznew = 0.0;
        for (int i = 0; i < n_local; i++)
        {
            rtznew += r_local[i] * z_local[i];
        }

        MPI_Allreduce(MPI_IN_PLACE, &rtznew, 1, MPI_DOUBLE, MPI_SUM, comm);

        if (fabs(rtzold) < 1e-14)
        {
            fprintf(stderr, "Error: Division by zero encountered in iteration %d.\n", k);
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
}