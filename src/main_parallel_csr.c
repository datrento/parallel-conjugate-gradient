#include "csr.h"
#include "utils.h"
#include "config.h"
#include "solver.h"
#include "solver_utils.h"
#include "csr_matrix_builder.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

int main(int argc, char *argv[])
{
    // MPI
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm comm = MPI_COMM_WORLD;
    // Problem size
    int grid_size = 200; // default grid size

    if (argc > 1)
    {
        grid_size = atoi(argv[1]);
    }

    // Validate grid size to prevent integer overflow
    if (validate_grid_size(grid_size, rank) != 0) // part of utils.c
    {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    double initial_guess = 0.0;
    int n = grid_size * grid_size * grid_size;

    // this never be triggered due to prior validation, but just in case
    if (n <= 0)
    {
        if (rank == 0)
        {
            fprintf(stderr, "[rank %d] Error: Computed matrix size n=%d is not positive. Check grid_size=%d for overflow.\n", rank, n, grid_size);
            fflush(stderr);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int base = n / size;
    int remainder = n % size;

    // flag to export csr matrix from rank 0 for verification of SPD properties
    int export_csr = 1;

    // Optimization parameters for Conjugate Gradient
    int max_iter = 1000;
    if (argc > 2)
        max_iter = atoi(argv[2]);
    double tol = 1e-10; // relative tolerance
    if (argc > 3)
        tol = atof(argv[3]);

    // Determine local number of rows and starting row for each process
    int n_local = (rank < remainder) ? base + 1 : base;
    int row_start = rank * base + (rank < remainder ? rank : remainder);

    // Pointers for glocal matrix and vectors
    double *x0 = NULL;
    double *p = NULL;

    // Allocate local vectors and matrix in each process
    double *diag_A_local = (double *)malloc(n_local * sizeof(double));
    double *b_local = (double *)malloc(n_local * sizeof(double));
    double *x_local = (double *)malloc(n_local * sizeof(double));
    double *p_local = (double *)malloc(n_local * sizeof(double));
    double *z_local = (double *)malloc(n_local * sizeof(double));
    double *Ap_local = (double *)malloc(n_local * sizeof(double));
    double *r_local = (double *)malloc(n_local * sizeof(double));

    if (!diag_A_local || !b_local || !x_local || !p_local || !z_local || !Ap_local || !r_local)
    {
        fprintf(stderr, "[rank %d] Error allocating local vectors(diag_A_local, b_local, x_local, p_local, z_local, Ap_local, r_local) of size %d\n", rank, n_local);
        fflush(stderr);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    // Local CSR matrix in each process
    CSR A_local;

    if (rank == 0)
    {
        const char *mtx_filename = "data/matrix_csr.mtx";

        printf("[rank %d]Building CSR matrix (7-point stencil) on rank 0...\n", rank);
        printf("[rank %d]Matrix size: %d x %d\n", rank, n, n);
        printf("[rank %d]Number of processes: %d\n", rank, size);
        printf("[rank %d]Each process local rows: %d\n", rank, n_local);
        printf("[rank %d] Grid size: %d x %d x %d\n", rank, grid_size, grid_size, grid_size);
        fflush(stdout);

        CSR A_full;

        if (csr_build_spd_full(&A_full, grid_size, rank) != 0)
        {
            fprintf(stderr, "[rank 0] Building full CSR matrix on rank 0\n");
            fflush(stderr);
            MPI_Abort(comm, EXIT_FAILURE);
        }

        // check SPD properties of the full matrix on rank 0
        // if (export_csr > 0)
        verify_spd_properties(&A_full, rank);

        // Debug: print memory usage of full CSR matrix on rank 0
        print_csr_memory_usage(rank, A_full.nnz, A_full.n, "full");

        // Distribute CSR matrix to all processes
        if (csr_distribute(&A_full, &A_local, n, n_local, row_start, rank, size, comm) != 0)
        {
            fprintf(stderr, "[rank 0] Error distributing CSR matrix from rank 0\n");
            fflush(stderr);
            MPI_Abort(comm, EXIT_FAILURE);
        }

        // export csr matrix to mtx for verification of SPD properties before freeing
        if (export_csr > 0)
            export_csr_to_mtx(mtx_filename, &A_full, n, rank);

        // free the full matrix in CSR format on rank 0
        csr_free(&A_full);
    }
    else
    {
        // Non-root ranks still need their local CSR slice
        if (csr_distribute(NULL, &A_local, n, n_local, row_start, rank, size, comm) != 0)
        {
            fprintf(stderr, "[rank %d] Error distributing CSR matrixd\n", rank);
            fflush(stderr);
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }

    // Allocate global vectors in each process
    p = (double *)malloc(n * sizeof(double)); // global p vector for Allgather

    if (!p)
    {
        fprintf(stderr, "[rank %d] Error allocating global vector p of size %d\n", rank, n);
        fflush(stderr);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    // Allocate x0 in each process (initial guess)
    x0 = (double *)malloc(n * sizeof(double));

    if (!x0)
    {
        fprintf(stderr, "[rank %d] Error allocating global vector x0 of size %d\n", rank, n);
        fflush(stderr);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    // Initialize x0 and b on rank 0
    for (int i = 0; i < n; i++)
    {
        x0[i] = initial_guess; // initial guess x0 as zero vector
    }

    // Broadcast x0 to all processes (initial guess) to allocate after freeing A_full on rank 0
    // MPI_Bcast(x0, n, MPI_DOUBLE, 0, comm); // initial guess

    // Build b = A * x_known where x_known is a vector of specific values (e.g., all 2.0)
    double *x_known = (double *)malloc(n * sizeof(double));

    if (!x_known)
    {
        fprintf(stderr, "[rank %d] Error allocating x_known of size %d\n", rank, n);
        fflush(stderr);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    // initialize x_known to all ranks
    for (int i = 0; i < n; i++)
    {
        x_known[i] = 2.0;
    }

    // b = A * x_known using local sparse matvec multiplication each process computes its local part
    // and stores in b_local
    csr_sparse_matvec_mult_local(&A_local, x_known, b_local);
    free(x_known);

    // get diagonal elements from local CSR matrix
    csr_get_diagonal_local(&A_local, diag_A_local, rank);

    // initialize local x from x0
    for (int i = 0; i < n_local; i++)
    {
        x_local[i] = x0[row_start + i];
    }

    // Precompute recvcounts and displs for variable-lenghth gatherv
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));

    if (!recvcounts || !displs)
    {
        fprintf(stderr, "[rank %d] Error allocating memory for recvcounts or displs\n", rank);
        fflush(stderr);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    if (rank == 0)
        printf("[rank %d] Precomputing recvcounts and displs for Allgatherv...\n", rank);

    // Rank order preserved while gathering n_local from all ranks
    MPI_Allgather(&n_local, 1, MPI_INT, recvcounts, 1, MPI_INT, comm);

    displs[0] = 0;
    for (int i = 1; i < size; i++)
    {
        displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    // Sanity: each rank's block starts at displs[rank] and has length recvcounts[rank]
    if (displs[rank] != row_start || recvcounts[rank] != n_local)
    {
        fprintf(stderr, "[rank %d]Rank %d: mismatch displs=%d recvcounts=%d vs row_start=%d n_local=%d\n",
                rank, rank, displs[rank], recvcounts[rank], row_start, n_local);
        fflush(stderr);
        MPI_Abort(comm, 1);
    }

    // Timing
    double local_time = 0.0;

    MPI_Barrier(comm); // synchronize before timing

    // Solve Ax = b using Jacobi Preconditioned Conjugate Gradient method
    local_time = MPI_Wtime();
    jacobi_preconditioned_conjugate_gradient(
        &A_local, diag_A_local, b_local, x0,
        x_local, r_local, p, p_local,
        z_local, Ap_local, n_local, n,
        recvcounts, displs,
        max_iter, tol, rank, comm);
    local_time = MPI_Wtime() - local_time;
    if (rank == 0)
    {
        printf("[rank %d] Completed Jacobi Preconditioned Conjugate Gradient solver.\n", rank);
        fflush(stdout);
    }

    //  Gather the local x (solution) to x0 in all processes
    MPI_Allgatherv(x_local, n_local, MPI_DOUBLE, x0, recvcounts, displs, MPI_DOUBLE, comm);

    // the minimum and maximum time taken among all processes
    double min_time = 0.0, max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, comm);

    // Verification the solutions
    verify_solution(&A_local, x0, b_local, Ap_local, n_local, rank, comm);

    if (rank == 0)
    {
        printf("[rank %d]Wall-clock time taken for Jacobi Preconditioned Conjugate Gradient: max %.6f s\n",
               rank, max_time);
        printf("[rank %d]Wall-clock time taken for Jacobi Preconditioned Conjugate Gradient: min %.6f s\n",
               rank, min_time);
        fflush(stdout);

        // store the max_time for performance analysis later
        const char *time_filename = "output/jcgtimes.txt";

        // Check if file exists to decide whether to write header
        int file_exists = 0;
        FILE *check_file = fopen(time_filename, "r");
        if (check_file)
        {
            file_exists = 1;
            fclose(check_file);
        }

        FILE *time_file = fopen(time_filename, "a");
        if (time_file)
        {
            // Write header if file is new
            if (!file_exists)
            {
                fprintf(time_file, "#grid_size num_processes max_time_seconds min_time_seconds\n");
            }

            // Append grid_size, number of processes, max and min time
            fprintf(time_file, "%d %d %.6f %.6f\n", grid_size, size, max_time, min_time);
            fclose(time_file);
            fprintf(stdout, "[rank %d]Appended time data to %s:  grid_size=%d, num_processes=%d, max_time=%.6f, min_time=%.6f\n", rank, time_filename, grid_size, size, max_time, min_time);
            fflush(stdout);
        }
        else
        {
            fprintf(stderr, "[rank %d] Error opening %s for writing\n", rank, time_filename);
            fflush(stderr);
        }
    }

    // Common frees (all ranks)
    free(x0);
    free(r_local);
    free(p_local);
    free(Ap_local);
    free(diag_A_local);
    free(z_local);
    free(x_local);
    free(b_local);
    free(p);
    free(recvcounts);
    free(displs);
    csr_free(&A_local);
    MPI_Finalize();
    return 0;
}
