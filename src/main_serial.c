#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void conjugate_gradient(double *A, double *b, double *x0, double *r, double *p, double *Ap, int n, int max_iter, double tol);

void generate_symmetric_positive_definite_matrix(int n);

int main(int argc, char *argv[])
{
    int n = 2;
    double start_time = 0.0, end_time = 0.0;
    double *A = (double *)malloc(n * n * sizeof(double));

    // allocate and initialize A
    A[0 * n + 0] = 4;
    A[0 * n + 1] = 1;
    A[1 * n + 0] = 1;
    A[1 * n + 1] = 3;

    double *b = (double *)malloc(n * sizeof(double));
    // allocate and initialize b
    b[0] = 1;
    b[1] = 2;

    double *x0 = (double *)malloc(n * sizeof(double));
    x0[0] = 2;
    x0[1] = 1;

    // initialize r, p, Ap
    double *r = (double *)malloc(n * sizeof(double));
    double *p = (double *)malloc(n * sizeof(double));
    double *Ap = (double *)malloc(n * sizeof(double));

    // Parameters for Conjugate Gradient
    int max_iter = 1000;
    double tol = 1e-10;

    start_time = time(NULL);
    conjugate_gradient(A, b, x0, r, p, Ap, n, max_iter, tol);
    end_time = (time(NULL) - start_time);

    double *x = x0;

    printf("Solution: ");
    for (int i = 0; i < n; i++)
    {
        printf("%f ", x[i]);
    }
    printf("\n");

    printf("Time taken: %lf seconds\n", end_time);
    free(A);
    free(b);
    free(x0);
    return 0;
}

void generate_symmetric_positive_definite_matrix(int n)
{
    // TODO : Implement matrix generation if needed
}

void conjugate_gradient(double *A, double *b, double *x0, double *r, double *p, double *Ap, int n, int max_iter, double tol)
{
    double alpha, beta, rsold, rsnew;

    // r0 = b - A * x0
    for (int i = 0; i < n; i++)
    {
        double Ax0_i = 0.0;
        for (int j = 0; j < n; j++)
        {
            Ax0_i += A[i * n + j] * x0[j];
        }

        r[i] = b[i] - Ax0_i;
        p[i] = r[i];
    }

    // rsold = r0' * r0
    rsold = 0.0;
    for (int i = 0; i < n; i++)
    {
        rsold += r[i] * r[i];
    }

    for (int iter = 0; iter < max_iter; iter++)
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

        // alpha = rsold / (p' * Ap)
        double pAp = 0.0;

        for (int i = 0; i < n; i++)
        {
            pAp += p[i] * Ap[i];
        }

        alpha = rsold / pAp;

        // x = x + alpha * p
        // Update x0 in place
        for (int i = 0; i < n; i++)
        {
            x0[i] += alpha * p[i];
        }

        // r = r - alpha * Ap
        for (int i = 0; i < n; i++)
        {
            r[i] -= alpha * Ap[i];
        }

        // rsnew = r' * r
        rsnew = 0.0;
        for (int i = 0; i < n; i++)
        {
            rsnew += r[i] * r[i];
        }

        // Check convergence
        if (sqrt(rsnew) < tol)
        {
            break;
        }

        // beta = rsnew / rsold
        beta = rsnew / rsold;

        // p = r + beta * p
        for (int i = 0; i < n; i++)
        {
            p[i] = r[i] + beta * p[i];
        }

        rsold = rsnew;
    }
}