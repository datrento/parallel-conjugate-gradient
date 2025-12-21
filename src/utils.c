#include <stdlib.h>
#include "utils.h"

void generate_symmetric_positive_definite_matrix(double *A, int n)
{
    /***
     * Generates a symmetric positive definite matrix A of size n x n.
     * For simplicity, we create a tridiagonal matrix which is known to be SPD.
     *
     * Parameters:
     *    A : Output matrix A (size: n x n)
     *    n : Size of the matrix
     *
     * Returns:
     *    None (A is updated in place)
     ***/
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
                A[i * n + j] = 2.0; // Diagonal elements
            else if (abs(i - j) == 1)

                A[i * n + j] = -1.0; // Off-diagonal elements

            else

                A[i * n + j] = 0.0; // Off-diagonal elements
        }
    }
}

void get_corresponding_b_and_x0(double *A, double *b, double *x0, int n)
{
    /***
     * Given a matrix A, this function generates a corresponding vector b
     * for a known solution x_known (which is a vector of ones) and initializes x0 to a zero vector.
     *
     * Parameters:
     *    A      : Symmetric positive definite matrix A (size: n x n)
     *    b      : Output vector b (size: n)
     *    x0     : Initial guess vector x0 (size: n)
     *    n      : Size of the matrix and vectors
     *
     * Returns:
     *    None (b and x0 are updated in place)
     ***/

    // known solution vector temporarily
    double *x_known = (double *)malloc(n * sizeof(double));

    // known solution x
    for (int i = 0; i < n; i++)
    {
        x_known[i] = 1.0; // for simplicity, let's assume the known solution is a vector of ones
        x0[i] = 0.0;      // initial guess x0 as zero vector
    }
    // initialize b
    for (int i = 0; i < n; i++)
    {
        b[i] = 0.0;
        for (int j = 0; j < n; j++)
        {
            b[i] += A[i * n + j] * x_known[j]; // b = A * [1, 1, ..., 1]^T
        }
    }

    free(x_known);
}
