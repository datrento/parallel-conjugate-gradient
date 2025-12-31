#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "dense_utils.h"
#include <math.h>

void jacobi_preconditioner_z(double *diag_A, double *r, double *z, int n);
void jacobi_preconditioned_conjugate_gradient(double *A, double *diag_A, double *b, double *x0, double *r, double *p, double *z, double *Ap, int n, int max_iter, double tol);
void get_diagonal_elements(double *A, double *diag_A, int n);
void vec_mat_mult(double *A, double *x, double *Ax, int n);
void verify_solution(double *A, double *Ax, double *x, double *b, int n);

int main(int argc, char *argv[])
{
    int grid_size = 10; // default grid size
    int n = grid_size * grid_size * grid_size;
    int need_spd_check = 1; // flag to export matrix for verification
    clock_t start_time = 0.0, end_time = 0.0;

    printf("Grid size: %d\n", grid_size);
    printf("Matrix size: %d x %d\n", n, n);

    // change A with dense format
    double *A = (double *)malloc(n * n * sizeof(double));
    double *diag_A = (double *)malloc(n * sizeof(double));

    double *b = (double *)malloc(n * sizeof(double));
    double *x0 = (double *)malloc(n * sizeof(double));

    // generate A for given n
    generate_symmetric_positive_definite_dense_matrix(A, grid_size);

    if (need_spd_check)
    {
        // export matrix to mtx file for verification
        export_dense_to_bin("../data/matrix.bin", A, n);
    }

    // get diagonal elements of A
    get_diagonal_elements(A, diag_A, n);

    // generate corresponding b for given A for a known solution x and initialize x0 to 0 vector
    get_corresponding_b_and_x0(A, b, x0, n);

    // initialize r, p, Ap
    double *r = (double *)malloc(n * sizeof(double));
    double *p = (double *)malloc(n * sizeof(double));
    double *Ap = (double *)malloc(n * sizeof(double));
    double *z = (double *)malloc(n * sizeof(double));

    // Parameters for Conjugate Gradient
    int max_iter = 1000;
    double tol = 1e-10;

    start_time = clock();
    jacobi_preconditioned_conjugate_gradient(A, diag_A, b, x0, r, p, z, Ap, n, max_iter, tol);
    end_time = clock() - start_time;

    // Reuse Ap for verification (will be overwritten inside verify_solution)
    verify_solution(A, Ap, x0, b, n);
    printf("Time taken: %.6f seconds\n", ((double)end_time) / CLOCKS_PER_SEC);
    free(A);
    free(b);
    free(x0);
    free(r);
    free(p);
    free(Ap);
    free(diag_A);
    free(z);
    return 0;
}

void vec_mat_mult(double *A, double *x, double *Ax, int n)
{
    /***
     * Multiplies matrix A with vector x and stores the result in vector Ax.
     *
     * Parameters:
     *    A_value  : Non-zero values of the matrix A in CRS format (size: nnz)
     *    A_row_ptr : Row pointers of the matrix A in CRS format (size: n + 1)
     *    A_col     : Column indices of the matrix A in CRS format (size: nnz)
     *    x  : Input vector x (size: n)
     *    Ax : Output vector Ax (size: n)
     *    n  : Size of the matrix and vectors
     *
     * Returns:
     *    None (Ax is updated in place)
     ***/
    for (int i = 0; i < n; i++)
    {
        Ax[i] = 0.0;
        for (int j = 0; j < n; j++)
        {
            Ax[i] += A[i * n + j] * x[j];
        }
    }
}

void get_diagonal_elements(double *A, double *diag_A, int n)
{
    /***
     * Retrieves the diagonal element A[i][i] from the CRS formatted matrix A.
     *
     * Parameters:
     *    A_value  : Non-zero values of the matrix A in CRS format (size: nnz)
     *    A_row_ptr : Row pointers of the matrix A in CRS format (size: n + 1)
     *    A_col     : Column indices of the matrix A in CRS format (size: nnz)
     *    i        : Row index for which to retrieve the diagonal element
     *
     * Returns:
     *    The diagonal element A[i][i]
     ***/
    for (int i = 0; i < n; i++)
    {
        diag_A[i] = 0.0;
        for (int j = 0; j < n; j++)
        {
            if (j == i)
            {
                diag_A[i] = A[i * n + j];
                break; // Exit loop once diagonal element is found
            }
        }

        if (diag_A[i] == 0.0)
        {
            fprintf(stderr, "Error: Diagonal element A[%d][%d] is zero.\n", i, i);
            exit(EXIT_FAILURE);
        }
    }
}
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

void verify_solution(double *A, double *Ax, double *x, double *b, int n)
{
    /***
     * Verifies the solution by computing Ax and comparing it to b.
     * Note: Ax buffer will be overwritten with A*x computation.
     ***/
    vec_mat_mult(A, x, Ax, n);

    double error = 0.0;
    for (int i = 0; i < n; i++)
    {
        double diff = Ax[i] - b[i];
        error += diff * diff;
    }
    error = sqrt(error);
    printf("Verification: ||Ax - b|| = %.6e\n", error);

    // sample solution output
    for (int i = 0; i < 10; i++)
    {
        printf("x[%d] = %.6e\n", i, x[i]);
    }
}

void jacobi_preconditioned_conjugate_gradient(double *A, double *diag_A, double *b, double *x0, double *r, double *p, double *z, double *Ap, int n, int max_iter, double tol)
{
    double alpha, beta, rsn0, rsnnew, rtzold, rtznew;

    // Ap = A * x0 use Ap as temporary storage
    vec_mat_mult(A, x0, Ap, n);

    // r0 = b - (Ax0 == Ap)
    for (int i = 0; i < n; i++)
    {
        r[i] = b[i] - Ap[i];
    }

    // z0 = M^-1 * r0 (Jacobi preconditioner) M^-1 = diag(A)^-1
    jacobi_preconditioner_z(diag_A, r, z, n);

    // p0 = z0
    for (int i = 0; i < n; i++)
    {
        p[i] = z[i];
    }

    // ||r0||^2
    rsn0 = 0.0;
    for (int i = 0; i < n; i++)
    {
        rsn0 += r[i] * r[i];
    }

    // rtz0 = r^T * z
    rtzold = 0.0;
    for (int i = 0; i < n; i++)
    {
        rtzold += r[i] * z[i];
    }

    // Main iteration loop
    for (int k = 0; k < max_iter; k++)
    {
        // Ap = A * p
        vec_mat_mult(A, p, Ap, n);

        // alpha = rtz / (p^T * Ap)
        double pAp = 0.0;
        for (int i = 0; i < n; i++)
        {
            pAp += p[i] * Ap[i];
        }

        if (pAp == 0.0)
        {
            fprintf(stderr, "Error: Division by zero encountered in iteration %d.\n", k);
            break; // Exit the loop to prevent division by zero
        }

        alpha = rtzold / pAp;

        // xk+1 = xk + alpha * pk
        for (int i = 0; i < n; i++)
        {
            x0[i] += alpha * p[i];
        }

        // rk+1 = rk - alpha * Apk
        for (int i = 0; i < n; i++)
        {
            r[i] -= alpha * Ap[i];
        }

        // rsnnew = r^T * r
        rsnnew = 0.0;
        for (int i = 0; i < n; i++)
        {
            rsnnew += r[i] * r[i];
        }

        // Check for convergence
        // for efficiency, we use the relative convergence criterion without computing square roots
        // ||rk+1|| < tol * ||r0||
        if (rsnnew < (tol * tol) * rsn0)
        {
            fprintf(stdout, "Converged in %d iterations.\n", k + 1);
            break;
        }

        // zk+1 = M^-1 * rk+1 (Jacobi preconditioner)
        jacobi_preconditioner_z(diag_A, r, z, n);

        // rtznew = r^T * z
        rtznew = 0.0;
        for (int i = 0; i < n; i++)
        {
            rtznew += r[i] * z[i];
        }

        // beta = rtznew / rtzold
        beta = rtznew / rtzold;

        // pk+1 = zk+1 + beta * pk
        for (int i = 0; i < n; i++)
        {
            p[i] = z[i] + beta * p[i];
        }

        // Update rtzold for next iteration
        rtzold = rtznew;
    }
}