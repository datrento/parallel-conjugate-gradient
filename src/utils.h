#ifndef UTILS_H
#define UTILS_H
// Functions for dense matrix generation and utilities
void export_dense_to_bin(const char *filename, double *A, int n);
void get_corresponding_b_and_x0(double *A, double *b, double *x0, int n);
void generate_symmetric_positive_definite_dense_matrix(double *A, int grid_size);

// Common utility functions
int validate_grid_size(int grid_size, int rank);
void compute_speedup_and_efficiency(const double *times, const int *procs, int n_runs);
#endif // UTILS_H