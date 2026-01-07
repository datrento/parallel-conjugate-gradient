#include "csr.h"
#include "utils.h"
#include "solver.h"
#include "solver_utils.h"
#include "csr_matrix_builder.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include "config.h"

__attribute__((unused)) static void mpi_gather_csr_to_root(CSR *A_local, CSR *A_global, MPI_Comm comm);

__attribute__((unused)) static void log_solver_performance_metrics(int rank, int size, int grid_size, double max_total_time, double min_total_time,
                                                                   double max_iter_time, double min_iter_time, int iters_done, double avg_iter_time);

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

    double known_solution_value = 2.0; // known solution value for verification

    // Set up 3D grid and z-slab decomposition
    Grid3D G;
    G.nx = grid_size;
    G.ny = grid_size;
    G.nz = grid_size;

    int base_nz = G.nz / size;                                          // base number of z-planes per process
    int remainder = G.nz % size;                                        // remainder to distribute among first 'remainder' ranks
    G.local_nz = base_nz + (rank < remainder ? 1 : 0);                  // local z-dimension size for this process (number of z-planes assigned to this process)
    G.z_start = rank * base_nz + (rank < remainder ? rank : remainder); // starting z-index for this process
    G.down = (rank > 0 ? rank - 1 : MPI_PROC_NULL);                     // below (lower rank) neighbor
    G.up = (rank < size - 1 ? rank + 1 : MPI_PROC_NULL);                // above (higher rank) neighbor

    // Optimization parameters for Conjugate Gradient
    int max_iter = 1000;
    if (argc > 2)
        max_iter = atoi(argv[2]);
    double tol = 1e-10; // relative tolerance
    if (argc > 3)
        tol = atof(argv[3]);

    // Determine local number of rows for each process
    int n_local = G.nx * G.ny * G.local_nz;

    // Allocate local vectors and matrix in each process
    double *b_local = (double *)malloc(n_local * sizeof(double));
    double *x_local = (double *)malloc(n_local * sizeof(double));
    double *Ax_local = (double *)malloc((n_local > 0 ? n_local : 1) * sizeof(double));

    if (!b_local || !x_local || !Ax_local)
    {
        fprintf(stderr, "[rank %d] Error allocating local vectors (b_local/x_local/Ax_local) of size %d\n", rank, n_local);
        fflush(stderr);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    // Local CSR matrix in each process
    CSR A_local = {0};

    if (rank == 0)
    {
        printf("[rank %d]Building local CSR (7-point) with z-slab partition...\n", rank);
        printf("[rank %d]Global grid: %d x %d x %d, local_nz=%d, z_start=%d\n", rank, G.nx, G.ny, G.nz, G.local_nz, G.z_start);
        fflush(stdout);
    }

    // Build local CSR matrix in each process based on its z-slab partition of the global grid
    build_local_matrix(&A_local, &G, rank);

#ifdef DEBUG
    // flag to export csr matrix from rank 0 for verification of SPD properties
    int export_csr = 0; // set to 1 to export the csr matrix to mtx file
    // Optionally export the full CSR matrix to Matrix Market format for verification
    if (export_csr && grid_size <= 20)
    {
        CSR A_full_gathered;

        // Gather to rank 0 the full matrix
        if (size > 1)
        {
            mpi_gather_csr_to_root(&A_local, &A_full_gathered, comm);
        }

        // Only rank 0 exports the file
        if (rank == 0)
        {
            CSR *A_full_ptr = (size > 1) ? &A_full_gathered : &A_local;
            char filename[256];
            snprintf(filename, sizeof(filename), "output/csr_matrix_full.mtx");
            export_csr_to_mtx(filename, A_full_ptr, A_full_ptr->n, rank);

            // Free gathered matrix if it was allocated
            if (size > 1)
            {
                csr_free(&A_full_gathered);
            }
        }
    }
#endif

    // Initialize x_local and b_local on each process
    for (int i = 0; i < n_local; i++)
    {
        x_local[i] = 0.0; // initial guess as zero vector
        b_local[i] = 0.0;
    }

    // b_local = A_local * x_known on each process and stores in b_local
    for (int i = 0; i < n_local; i++)
    {
        double row_sum = 0.0;
        for (int j = A_local.row_ptr[i]; j < A_local.row_ptr[i + 1]; j++)
        {
            row_sum += A_local.values[j];
        }
        b_local[i] = row_sum * known_solution_value; // b_local = A_local * x_known_local
    }

    // Timing metrics container returned from solver
    cg_metrics_t metrics = {0};
// This is controlled by the Makefile (-DBASE_LINE_ONLY=1 or 0)
#if (BASE_LINE_ONLY == 1)
    if (rank == 0)
    {
        printf("-------------------------------------------------------\n");
        printf("[rank %d][SOLVER] Mode: BASELINE Jacobi-Preconditioned CG\n", rank);
        printf("[rank %d][COMM]   Method: Global Allgatherv\n", rank);
        printf("[rank %d] Number of processes: %d\n", rank, size);
        printf("-------------------------------------------------------\n");
        fflush(stdout);
    }
    jacobi_preconditioned_cg(&A_local, &G, b_local, x_local, max_iter, tol, &metrics);
#else
    if (rank == 0)
    {
        printf("-------------------------------------------------------\n");
        printf("[rank %d][SOLVER] Mode: PIPELINED Jacobi-CG\n", rank);
        printf("[rank %d] Number of processes: %d\n", rank, size);
        printf("[rank %d][COMM]   Method: Non-blocking P2P Halo Exchange\n", rank);
        printf("-------------------------------------------------------\n");
        fflush(stdout);
    }
    jacobi_preconditioned_pipelined_cg(&A_local, &G, b_local, x_local, max_iter, tol, &metrics);
#endif

    double iter_time_local = metrics.iter_time;
    double total_time_local = metrics.total_time;
    int iters_done = metrics.iters;

    int plane_size = G.nx * G.ny;

    // Halo buffers
    double *r_up = malloc(plane_size * sizeof(double));   // halo buffer for upper neighbor
    double *r_down = malloc(plane_size * sizeof(double)); // halo buffer for lower neighbor

    // compute Ax_local = A_local * x_local with halo exchanges for verification
    MPI_Request h_req[4];

    if (!r_up || !r_down)
    {
        fprintf(stderr, "[rank %d] Error: Memory allocation failed for halo buffers r_up or r_down\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Initial halo exchange for x_local
    halo_exchange_init(&G, x_local, r_up, r_down, h_req);

    // SpMV: w0 = A * u0 (internal + boundary)
    csr_sparse_matvec_mult_internal(&A_local, &G, x_local, Ax_local);
    MPI_Waitall(4, h_req, MPI_STATUSES_IGNORE);
    csr_sparse_matvec_mult_boundary(&A_local, &G, x_local, r_up, r_down, Ax_local);

    MPI_Request gather_req[2];
    // the minimum and maximum time taken among all processes (iteration-only and total)
    double local_times[2] = {total_time_local, iter_time_local};
    double min_times[2];
    double max_times[2];

    MPI_Ireduce(local_times, min_times, 2, MPI_DOUBLE, MPI_MIN, 0, comm, &gather_req[0]);
    MPI_Ireduce(local_times, max_times, 2, MPI_DOUBLE, MPI_MAX, 0, comm, &gather_req[1]);

    // Wait for all non-blocking operations to complete
    MPI_Waitall(2, gather_req, MPI_STATUSES_IGNORE);
    double min_total_time = 0, min_iter_time = 0;
    double max_total_time = 0, max_iter_time = 0;
    double avg_iter_time = 0;
    if (rank == 0)
    {
        min_total_time = min_times[0];
        min_iter_time = min_times[1];
        max_total_time = max_times[0];
        max_iter_time = max_times[1];
        avg_iter_time = max_iter_time / iters_done;

        log_solver_performance_metrics(rank, size, grid_size,
                                       max_total_time, min_total_time,
                                       max_iter_time, min_iter_time, iters_done, avg_iter_time);
    }
    // Verification the solutions
    verify_solution(Ax_local, b_local, n_local, rank, comm);

    // Common frees (all ranks)
    free(r_up);
    free(r_down);
    free(Ax_local);
    free(x_local);
    free(b_local);
    csr_free(&A_local);
    MPI_Finalize();
    return 0;
}

__attribute__((unused)) static void mpi_gather_csr_to_root(CSR *A_local, CSR *A_global, MPI_Comm comm)
{
    /**
     * Gather the full CSR matrix from all processes to rank 0.
     * Parameters:
     *      A_local: local CSR matrix
     *      A_global: pointer to full CSR matrix (only allocated on rank 0)
     *      comm: MPI communicator
     * Returns: void
     **/
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int *nnz_counts = NULL;
    int *row_counts = NULL;
    if (rank == 0)
    {
        nnz_counts = (int *)malloc(size * sizeof(int));
        row_counts = (int *)malloc(size * sizeof(int));
    }

    int local_nnz = A_local->nnz;
    int local_rows = A_local->n_local;

    int *row_ptr_counts = NULL;
    int *row_ptr_displs = NULL;
    int *nnz_displs = NULL;

    MPI_Gather(&local_nnz, 1, MPI_INT, nnz_counts, 1, MPI_INT, 0, comm);
    MPI_Gather(&local_rows, 1, MPI_INT, row_counts, 1, MPI_INT, 0, comm);

    if (rank == 0)
    {
        // Compute total nnz and rows
        int total_nnz = 0;
        int total_rows = 0;
        for (int i = 0; i < size; i++)
        {
            total_nnz += nnz_counts[i];
            total_rows += row_counts[i];
        }

        // Allocate global CSR matrix
        // csr_init(A_global, total_rows, total_nnz);
        A_global->n = total_rows;
        A_global->n_local = total_rows;
        A_global->row_start = 0;
        A_global->nnz = total_nnz;
        A_global->row_ptr = (int *)malloc((total_rows + 1) * sizeof(int));
        A_global->col_indices = (int *)malloc(total_nnz * sizeof(int));
        A_global->values = (double *)malloc(total_nnz * sizeof(double));
        A_global->inv_diag = NULL; // not needed for full matrix

        // Prepare row_ptr counts and displacements (only on rank 0)
        row_ptr_counts = (int *)malloc(size * sizeof(int));
        row_ptr_displs = (int *)malloc(size * sizeof(int));
        row_ptr_counts[0] = row_counts[0] + 1; // Include row_ptr[0] = 0 from rank 0
        for (int i = 1; i < size; i++)
        {
            row_ptr_counts[i] = row_counts[i]; // Skip duplicate row_ptr[0] from other ranks
        }
        row_ptr_displs[0] = 0;
        for (int i = 1; i < size; i++)
        {
            row_ptr_displs[i] = row_ptr_displs[i - 1] + row_ptr_counts[i - 1];
        }

        // Prepare nnz displacements
        nnz_displs = (int *)malloc(size * sizeof(int));
        nnz_displs[0] = 0;
        for (int i = 1; i < size; i++)
        {
            nnz_displs[i] = nnz_displs[i - 1] + nnz_counts[i - 1];
        }
    }

    // ALL ranks participate in Gatherv operations
    // Rank 0 sends all local_rows + 1 elements, others skip first element
    int send_count = (rank == 0) ? local_rows + 1 : local_rows;
    int *send_buf = (rank == 0) ? A_local->row_ptr : A_local->row_ptr + 1;

    MPI_Gatherv(send_buf, send_count, MPI_INT,
                A_global->row_ptr, row_ptr_counts, row_ptr_displs,
                MPI_INT, 0, comm);

    // ALL ranks participate in MPI_Gatherv operations
    MPI_Gatherv(A_local->col_indices, local_nnz, MPI_INT,
                A_global->col_indices, nnz_counts, nnz_displs,
                MPI_INT, 0, comm);

    MPI_Gatherv(A_local->values, local_nnz, MPI_DOUBLE,
                A_global->values, nnz_counts, nnz_displs,
                MPI_DOUBLE, 0, comm);

    // Only rank 0 needs to adjust offsets and clean up
    if (rank == 0)
    {
        // Adjust row_ptr offsets
        for (int i = 1; i < size; i++)
        {
            int offset = 0;
            for (int j = 0; j < i; j++)
            {
                offset += nnz_counts[j];
            }
            for (int k = row_ptr_displs[i]; k < row_ptr_displs[i] + row_counts[i]; k++)
            {
                A_global->row_ptr[k] += offset;
            }
        }

        free(nnz_counts);
        free(row_counts);
        free(row_ptr_counts);
        free(row_ptr_displs);
        free(nnz_displs);
    }
}

__attribute__((unused)) static void log_solver_performance_metrics(
    int rank, int size, int grid_size,
    double max_total_time, double min_total_time,
    double max_iter_time, double min_iter_time, int iters_done, double avg_iter_time)
{
    /**
     * Log the running time metrics to stdout and to a file for performance analysis.
     * The file "output/jcgtimes.txt" will contain the grid size, number of processes,
     * max and min total time, max and min iteration time, and number of iterations performed.
     * Parameters:
     *     rank: MPI rank of the calling process
     *     size: Total number of MPI processes
     *     grid_size: Size of the 3D grid (assumed cubic)
     *     max_total_time: Maximum total time taken among all processes
     *     min_total_time: Minimum total time taken among all processes
     *     max_iter_time: Maximum iteration-only time taken among all processes
     *     min_iter_time: Minimum iteration-only time taken among all processes
     *     iters_done: Number of iterations performed
     * Returns: None
     */
    printf("[rank %d]Algorithm (total solver call) time: max %.6f => min %.6f s\n", rank, max_total_time, min_total_time);
    printf("[rank %d]Algorithm (only loop) time: max %.6f => min %.6f s\n", rank, max_iter_time, min_iter_time);
    printf("[rank %d]Iterations performed: %d\n", rank, iters_done);
    printf("[rank %d]Average time per iteration (max): %.6f s\n", rank, avg_iter_time);
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
            fprintf(time_file, "#grid_size num_processes total_time_max total_time_min iter_time_max iter_time_min total_iters avg_iter_time\n");
        }

        // Append grid_size, number of processes, max and min time
        fprintf(time_file, "%d %d %.6f %.6f %.6f %.6f %d %.6f\n", grid_size, size, max_total_time, min_total_time, max_iter_time, min_iter_time, iters_done, avg_iter_time);
        fclose(time_file);
        fprintf(stdout, "[rank %d]Appended time data to %s:  grid_size=%d, num_processes=%d, total_time_max=%.6f, total_time_min=%.6f, iter_time_max=%.6f, iter_time_min=%.6f, iters_done=%d\n",
                rank, time_filename, grid_size, size, max_total_time, min_total_time, max_iter_time, min_iter_time, iters_done);
        fflush(stdout);
    }
    else
    {
        fprintf(stderr, "[rank %d] Error opening %s for writing\n", rank, time_filename);
        fflush(stderr);
    }
}