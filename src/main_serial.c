#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "utils.h"

void jacobi_pre_conditioned_conjugate_gradient(double *A, double *b, double *x0, double *r, double *p, double *z, double *Ap, int n, int max_iter, double tol);

int main(int argc, char *argv[])
{
    int n = 2;
    double start_time = 0.0, end_time = 0.0;
    double *A = (double *)malloc(n * n * sizeof(double));
    double *b = (double *)malloc(n * sizeof(double));
    double *x0 = (double *)malloc(n * sizeof(double));

    // generate A for given n
    generate_symmetric_positive_definite_matrix(A, n);

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

    start_time = time(NULL);
    jacobi_pre_conditioned_conjugate_gradient(A, b, x0, r, p, z, Ap, n, max_iter, tol);
    end_time = (time(NULL) - start_time);

    printf("Solution: ");
    for (int i = 0; i < n; i++)
    {
        printf("%f ", x0[i]);
    }
    printf("\n");

    printf("Time taken: %lf seconds\n", end_time);
    free(A);
    free(b);
    free(x0);
    free(r);
    free(p);
    free(Ap);
    free(z);
    return 0;
}

void jacobi_pre_conditioned_conjugate_gradient(double *A, double *b, double *x0, double *r, double *p, double *z, double *Ap, int n, int max_iter, double tol)
{
    double alpha, beta, rsn0, rsnnew, rtzold, rtznew;

    // r0 = b - A * x0
    for (int i = 0; i < n; i++)
    {
        double Ax0_i = 0.0;
        for (int j = 0; j < n; j++)
        {
            Ax0_i += A[i * n + j] * x0[j];
        }

        r[i] = b[i] - Ax0_i;
    }

    // z0 = M^-1 * r0 (Jacobi preconditioner) M^-1 = diag(A)^-1
    // p0 = z0
    for (int i = 0; i < n; i++)
    {
        z[i] = r[i] / A[i * n + i];
        p[i] = z[i];
    }

    // Compute initial residual norm
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
        for (int i = 0; i < n; i++)
        {
            Ap[i] = 0.0;
            for (int j = 0; j < n; j++)
            {
                Ap[i] += A[i * n + j] * p[j];
            }
        }

        // alpha = rtz / (p^T * Ap)
        double pAp = 0.0;
        for (int i = 0; i < n; i++)
        {
            pAp += p[i] * Ap[i];
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
        if (sqrt(rsnnew) < tol * sqrt(rsn0))
        {
            break;
        }

        // zk+1 = M^-1 * rk+1
        for (int i = 0; i < n; i++)
        {
            z[i] = r[i] / A[i * n + i];
        }

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

        rtzold = rtznew;
    }
}