#pragma once
// Functions for dense matrix generation and utilities
void export_dense_to_bin(const char *filename, double *A, int n);
void get_corresponding_b_and_x0(double *A, double *b, double *x0, int n);
void generate_symmetric_positive_definite_dense_matrix(double *A, int grid_size);