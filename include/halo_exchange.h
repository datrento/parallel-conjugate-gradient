#pragma once
#include <mpi.h>
#include "csr.h"

// Structure to hold halo/ghost exchange information
typedef struct
{
    int n_neighbors;     // Number of processes we communicate with
    int *neighbor_ranks; // Array of neighbor rank IDs [n_neighbors]

    int *send_counts; // How many values to send to each neighbor [n_neighbors]
    int *recv_counts; // How many values to receive from each neighbor [n_neighbors]

    int *send_offsets; // Offset into send_indices for each neighbor [n_neighbors+1] (CSR-style)
    int *recv_offsets; // Offset into recv_indices for each neighbor [n_neighbors+1] (CSR-style)

    int *send_indices; // Local indices to send to neighbors [total_send_size]
    int *recv_indices; // Global column indices to receive from neighbors [total_recv_size]

    int total_ghost_size; // Total number of ghost/halo values we receive (same as recv_offsets[n_neighbors])
    double *ghost_values; // Buffer to store received ghost values [total_ghost_size]
    int *ghost_to_global; // Map:  ghost_index -> global column index [total_ghost_size]

} HaloExchange;

// Initialize and build the halo exchange structure by analyzing CSR sparsity pattern
int halo_exchange_build(HaloExchange *halo, const CSR *A_local,
                        int n_local, int row_start, int row_end,
                        int rank, int size, int n, MPI_Comm comm);

// Free halo exchange structure
void halo_exchange_free(HaloExchange *halo);

int halo_exchange_execute(HaloExchange *halo, const double *p_local, MPI_Comm comm);