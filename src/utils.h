#ifndef UTILS_H
#define UTILS_H
void generate_symmetric_positive_definite_matrix(double *A, int n);
void get_corresponding_b_and_x0(double *A, double *b, double *x0, int n);
#endif // UTILS_H