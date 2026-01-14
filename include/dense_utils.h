#pragma once
// Functions for dense matrix generation and utilities
/**
 * This utility functions used for dense matrix generation, exporting, and
 * generating corresponding vectors. which used for the dense variant of HPCG.
 * we don't use these functions in the main HPCG benchmark since on the first test the memory requirements makes it impractical.
 * But these functions are provided for completeness and for users who want to experiment with dense matrices.
 */

/**
 * Exports a dense matrix A to a binary file.
 * Parameters:
 *    filename: output file name
 *    A: pointer to dense matrix (row-major)
 *    n: matrix size (n x n)
 * Returns: void
 */
void export_dense_to_bin(const char *filename, double *A, int n);

/**
 * Given a dense matrix A, this function generates a corresponding vector b
 * for a known solution x_known (which is a vector of twos) and initializes x0 to a zero vector.
 *
 * Parameters:
 *    A : Input matrix A in dense format (size: n × n, stored row-major)
 *    b      : Output vector b (size: n)
 *    x0     : Initial guess vector x0 (size: n)
 *    n      : Size of the matrix and vectors
 *
 * Returns:
 *    None (b and x0 are updated in place)
 */
void get_corresponding_b_and_x0(double *A, double *b, double *x0, int n);

/**
 * Generates a symmetric positive definite dense matrix.
 *
 * Parameters:
 *    A         : Output matrix A in dense format (size: n × n, stored row-major)
 *    grid_size : Size of the matrix (n)
 *
 * Returns:
 *    None (A is updated in place)
 */
void generate_symmetric_positive_definite_dense_matrix(double *A, int grid_size);