#ifndef UTILS_H
#define UTILS_H
void generate_symmetric_positive_definite_dense_matrix(double *A, int grid_size);
void get_corresponding_b_and_x0(double *A, double *b, double *x0, int n);
void export_dense_to_bin(const char *filename, double *A, int n);
#endif // UTILS_H