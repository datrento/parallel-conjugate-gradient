#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "utils.h"

void conjugate_gradient(double *A_local, double *b_local, double *x_local, double *x0,
                        double *r_local, double *p_local, double *Ap_local, double *p,
                        int n_local, int n, int max_iter, double tol,
                        MPI_Comm comm);

int main(int argc, char *argv[])
{
    // Timing
    double start_time = 0.0, end_time = 0.0, sol = 1.0;

    // MPI
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Parameters for Conjugate Gradient
    int n = 10000;       // Size of the matrix (n x n)
    int max_iter = 1000; // Maximum number of iterations
    double tol = 1e-8;   // Tolerance for convergence

    // Check if n is divisible by size
    if (n % size != 0)
    {
        if (rank == 0)
            fprintf(stderr, "Matrix size n must be divisible by number of processes.\n");
        MPI_Abort(comm, EXIT_FAILURE);
    }

    int n_local = n / size; // Number of rows per process (assuming n is divisible by size) of A

    // Pointer declarations(set to NULL for safety)
    double *A = NULL;
    double *b = NULL;
    double *x0 = NULL;
    double *p = NULL; // Global search direction vector
    double *p_local = NULL;
    double *r_local = NULL;
    double *Ap_local = NULL;
    double *A_local = NULL;
    double *b_local = NULL;
    double *x_local = NULL;

    // Allocate main local portions
    A_local = (double *)malloc(n_local * n * sizeof(double));
    x_local = (double *)malloc(n_local * sizeof(double));
    b_local = (double *)malloc(n_local * sizeof(double));

    if (rank == 0)
    {
        A = (double *)malloc(n * n * sizeof(double));
        b = (double *)malloc(n * sizeof(double));
        x0 = (double *)malloc(n * sizeof(double));
        // initialize A
        generate_symmetric_positive_definite_matrix(A, n);

        // initialize b and x0 (set to zero vector)
        get_corresponding_b_and_x0(A, b, x0, n);
    }
    else
    {
        // Allocate local portions for other processes
        x0 = (double *)malloc(n * sizeof(double)); // TODO: optimize this
    }

    // Distribute A and b to all processes
    MPI_Scatter(A, n_local * n, MPI_DOUBLE, A_local, n_local * n, MPI_DOUBLE, 0, comm);
    MPI_Scatter(b, n_local, MPI_DOUBLE, b_local, n_local, MPI_DOUBLE, 0, comm);

    // Broadcast initial guess x0 to all processes, then set local slice x_local
    MPI_Bcast(x0, n, MPI_DOUBLE, 0, comm); // TODO: optimize this
    for (int i = 0; i < n_local; i++)
    {
        x_local[i] = x0[i + rank * n_local];
    }

    // Allocate local vectors
    r_local = (double *)malloc(n_local * sizeof(double));
    p_local = (double *)malloc(n_local * sizeof(double));
    Ap_local = (double *)malloc(n_local * sizeof(double));

    // Global search direction vector
    p = (double *)malloc(n * sizeof(double)); // TODO: need a little bit of optimization here if possible

    MPI_Barrier(comm);

    start_time = MPI_Wtime();
    conjugate_gradient(A_local, b_local, x_local, x0,
                       r_local, p_local, Ap_local, p,
                       n_local, n, max_iter, tol,
                       comm);
    end_time = (MPI_Wtime() - start_time);

    double *x = NULL; // TODO: if there is a better way to gather the result without consuming extra memory
    if (rank == 0)
    {
        x = (double *)malloc(n * sizeof(double));
    }

    MPI_Gather(x_local, n_local, MPI_DOUBLE, x, n_local, MPI_DOUBLE, 0, comm);

    if (rank == 0)
    {
        printf("Solution: ");
        for (int i = 0; i < n; i++)
        {
            if (round(x[i]) != sol)
            {
                printf("Error at index %d: Expected %f, Got %f\n", i, sol, x[i]);
                break;
            }
            else if (i == n - 1)
                printf("The solution values are correct.\n");
        }
        printf("\n");
    }

    double min_time = 0.0, max_time = 0.0;
    MPI_Reduce(&end_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&end_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, comm);

    if (rank == 0)
    {
        printf("Maximum Time taken among processes: %lf seconds\n", max_time);
        printf("Minimum Time taken among processes: %lf seconds\n", min_time);
    }

    free(A);
    free(b);
    free(x0);
    free(x);
    free(p);
    free(r_local);
    free(p_local);
    free(Ap_local);
    free(A_local);
    free(b_local);
    free(x_local);

    MPI_Finalize();
    return 0;
}

void conjugate_gradient(
    double *A_local, double *b_local, double *x_local, double *x0,
    double *r_local, double *p_local, double *Ap_local, double *p,
    int n_local, int n, int max_iter, double tol,
    MPI_Comm comm)
{
    /***
     *
     * Conjugate Gradient Method to solve Ax = b
     *
     * Parameters:
     *    A_local : Local portion of the matrix A for each process (row-wise partitioned) (size: n_local x n)
     *    b_local : Local portion of the vector b for each process (size: n_local)
     *    x0      : Initial guess for the solution vector x (size: n)
     *    r_local : Local portion of the residual vector r (size: n_local)
     *    p_local : Local portion of the search direction vector p (size: n_local
     *    Ap_local: Local portion of the matrix-vector product Ap (size: n_local)
     *    p       : Global search direction vector p (size: n)
     *    n_local  : Number of rows assigned to each process
     *    n        : Total size of the matrix/vector
     *    max_iter : Maximum number of iterations
     *    tol      : Tolerance for convergence
     *    comm     : MPI communicator
     * Returns:
     *    None (the solution is updated in place in x0)
     *
     ***/

    // TODO: try to opimize the loop per process, more with OPENMP if possible

    // rsold and rsnew are used for both local and global values to save memory(in place reduction)
    // alpha and beta are only global values
    double alpha, beta, rsold, rsnew;

    // r0 = b - A * x0
    for (int i = 0; i < n_local; i++)
    {
        double Ax0_i = 0.0;
        for (int j = 0; j < n; j++)
        {
            Ax0_i += A_local[i * n + j] * x0[j];
        }

        r_local[i] = b_local[i] - Ax0_i;
        p_local[i] = r_local[i];
    }

    // rsold = r0' * r0
    rsold = 0.0;
    for (int i = 0; i < n_local; i++)
    {
        rsold += r_local[i] * r_local[i];
    }

    // Reduce rsold across all processes and store in rsold to save memory
    MPI_Allreduce(MPI_IN_PLACE, &rsold, 1, MPI_DOUBLE, MPI_SUM, comm);

    double rs0 = rsold; // save initial ||r0||^2 for relative test

    for (int iter = 0; iter < max_iter; iter++)
    {
        // Compute global p vector from all local p_local
        MPI_Allgather(p_local, n_local, MPI_DOUBLE, p, n_local, MPI_DOUBLE, comm);

        // Ap_local = A_local * p
        for (int i = 0; i < n_local; i++)
        {
            Ap_local[i] = 0.0;

            for (int j = 0; j < n; j++)
            {
                Ap_local[i] += A_local[i * n + j] * p[j];
            }
        }

        // alpha = rsold / (p_local' * Ap_local)
        double pAp = 0.0;

        for (int i = 0; i < n_local; i++)
        {
            pAp += p_local[i] * Ap_local[i];
        }

        // Reduce pAp across all processes
        MPI_Allreduce(MPI_IN_PLACE, &pAp, 1, MPI_DOUBLE, MPI_SUM, comm);

        // global value
        alpha = rsold / pAp;

        // x = x + alpha * p
        // Update x0 in place
        for (int i = 0; i < n_local; i++)
        {
            x_local[i] += alpha * p_local[i];
        }

        // r = r - alpha * Ap
        for (int i = 0; i < n_local; i++)
        {
            r_local[i] -= alpha * Ap_local[i];
        }

        // rsnew = r' * r
        rsnew = 0.0;
        for (int i = 0; i < n_local; i++)
        {
            rsnew += r_local[i] * r_local[i];
        }

        MPI_Allreduce(MPI_IN_PLACE, &rsnew, 1, MPI_DOUBLE, MPI_SUM, comm);

        // Check convergence using relative residual to r0
        if (sqrt(rsnew / rs0) < tol)
        {
            break;
        }

        // beta = rsnew / rsold
        beta = rsnew / rsold;

        // p = r + beta * p
        for (int i = 0; i < n_local; i++)
        {
            p_local[i] = r_local[i] + beta * p_local[i];
        }

        rsold = rsnew;
    }
}