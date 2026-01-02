#include "halo_exchange.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// Add this helper function at the top of the file, after the includes

static int find_column_owner(int col, int size, int n)
{
    /***
     * Find which process owns a given column index.
     * Uses same distribution logic as row distribution.
     ***/
    int base = n / size;
    int remainder = n % size;

    if (col < (base + 1) * remainder)
    {
        return col / (base + 1);
    }
    else
    {
        return remainder + (col - (base + 1) * remainder) / base;
    }
}
void halo_exchange_free(HaloExchange *halo)
{
    /***
     * Frees all memory allocated in the HaloExchange structure.
     *
     * Parameters:
     *      halo:  pointer to HaloExchange structure
     * Returns:  void
     ***/

    if (halo->neighbor_ranks != NULL)
    {
        free(halo->neighbor_ranks);
        halo->neighbor_ranks = NULL;
    }

    if (halo->send_counts != NULL)
    {
        free(halo->send_counts);
        halo->send_counts = NULL;
    }

    if (halo->recv_counts != NULL)
    {
        free(halo->recv_counts);
        halo->recv_counts = NULL;
    }

    if (halo->send_offsets != NULL)
    {
        free(halo->send_offsets);
        halo->send_offsets = NULL;
    }

    if (halo->recv_offsets != NULL)
    {
        free(halo->recv_offsets);
        halo->recv_offsets = NULL;
    }

    if (halo->send_indices != NULL)
    {
        free(halo->send_indices);
        halo->send_indices = NULL;
    }

    if (halo->recv_indices != NULL)
    {
        free(halo->recv_indices);
        halo->recv_indices = NULL;
    }

    if (halo->ghost_values != NULL)
    {
        free(halo->ghost_values);
        halo->ghost_values = NULL;
    }

    if (halo->ghost_to_global != NULL)
    {
        free(halo->ghost_to_global);
        halo->ghost_to_global = NULL;
    }

    halo->n_neighbors = 0;
    halo->total_ghost_size = 0;
}

int halo_exchange_build(HaloExchange *halo, const CSR *A_local,
                        int n_local, int row_start, int row_end,
                        int rank, int size, int n, MPI_Comm comm)
{
    /***
     * Builds the halo exchange structure by analyzing CSR matrix sparsity pattern.
     ***/

    // Initialize all pointers to NULL
    halo->neighbor_ranks = NULL;
    halo->send_counts = NULL;
    halo->recv_counts = NULL;
    halo->send_offsets = NULL;
    halo->recv_offsets = NULL;
    halo->send_indices = NULL;
    halo->recv_indices = NULL;
    halo->ghost_values = NULL;
    halo->ghost_to_global = NULL;
    halo->n_neighbors = 0;
    halo->total_ghost_size = 0;

    // Step 1: Count how many columns we need from each process
    int *cols_needed_from_process = (int *)calloc(size, sizeof(int));
    if (!cols_needed_from_process)
    {
        fprintf(stderr, "[rank %d] Error allocating cols_needed_from_process\n", rank);
        return -1;
    }

    // Scan all non-zeros to find remote columns
    for (int i = 0; i < n_local; i++)
    {
        for (int j = A_local->row_ptr[i]; j < A_local->row_ptr[i + 1]; j++)
        {
            int col = A_local->col_indices[j];

            // Check if column is remote
            if (col < row_start || col >= row_end)
            {
                int owner = find_column_owner(col, size, n);
                cols_needed_from_process[owner]++;
            }
        }
    }

    // Step 2: Determine which processes are neighbors (have non-zero counts)
    // Count neighbors
    int n_neighbors = 0;
    for (int p = 0; p < size; p++)
    {
        if (cols_needed_from_process[p] > 0)
        {
            n_neighbors++;
        }
    }

    halo->n_neighbors = n_neighbors;

    if (n_neighbors == 0)
    {
        // No remote communication needed
        free(cols_needed_from_process);
        return 0;
    }

    // Step 3: Allocate arrays for neighbors
    halo->neighbor_ranks = (int *)malloc(n_neighbors * sizeof(int));
    halo->recv_counts = (int *)malloc(n_neighbors * sizeof(int));
    halo->recv_offsets = (int *)malloc((n_neighbors + 1) * sizeof(int));

    if (!halo->neighbor_ranks || !halo->recv_counts || !halo->recv_offsets)
    {
        fprintf(stderr, "[rank %d] Error allocating neighbor arrays\n", rank);
        free(cols_needed_from_process);
        halo_exchange_free(halo);
        return -1;
    }

    // Fill neighbor_ranks and recv_counts
    int neighbor_idx = 0;
    for (int p = 0; p < size; p++)
    {
        if (cols_needed_from_process[p] > 0)
        {
            halo->neighbor_ranks[neighbor_idx] = p;
            halo->recv_counts[neighbor_idx] = cols_needed_from_process[p];
            neighbor_idx++;
        }
    }

    // Build recv_offsets (CSR-style)
    halo->recv_offsets[0] = 0;
    for (int i = 0; i < n_neighbors; i++)
    {
        halo->recv_offsets[i + 1] = halo->recv_offsets[i] + halo->recv_counts[i];
    }
    halo->total_ghost_size = halo->recv_offsets[n_neighbors];

    // Step 4: Collect unique column indices for each neighbor

    // Allocate temporary storage to collect columns per neighbor (with potential duplicates)
    int **temp_cols_per_neighbor = (int **)malloc(n_neighbors * sizeof(int *));
    int *temp_counts = (int *)calloc(n_neighbors, sizeof(int));

    if (!temp_cols_per_neighbor || !temp_counts)
    {
        fprintf(stderr, "[rank %d] Error allocating temporary storage\n", rank);
        free(cols_needed_from_process);
        halo_exchange_free(halo);
        return -1;
    }

    // Allocate space for each neighbor (upper bound with duplicates)
    for (int i = 0; i < n_neighbors; i++)
    {
        temp_cols_per_neighbor[i] = (int *)malloc(halo->recv_counts[i] * sizeof(int));
        if (!temp_cols_per_neighbor[i])
        {
            fprintf(stderr, "[rank %d] Error allocating temp columns for neighbor %d\n", rank, i);
            for (int k = 0; k < i; k++)
            {
                free(temp_cols_per_neighbor[k]);
            }
            free(temp_cols_per_neighbor);
            free(temp_counts);
            free(cols_needed_from_process);
            halo_exchange_free(halo);
            return -1;
        }
    }

    // Scan matrix and collect columns (with duplicates)
    for (int i = 0; i < n_local; i++)
    {
        for (int j = A_local->row_ptr[i]; j < A_local->row_ptr[i + 1]; j++)
        {
            int col = A_local->col_indices[j];

            if (col < row_start || col >= row_end)
            {
                int owner = find_column_owner(col, size, n);

                // Find neighbor index
                int neighbor_idx = -1;
                for (int k = 0; k < n_neighbors; k++)
                {
                    if (halo->neighbor_ranks[k] == owner)
                    {
                        neighbor_idx = k;
                        break;
                    }
                }

                // Check if column already exists for this neighbor
                int already_exists = 0;
                for (int k = 0; k < temp_counts[neighbor_idx]; k++)
                {
                    if (temp_cols_per_neighbor[neighbor_idx][k] == col)
                    {
                        already_exists = 1;
                        break;
                    }
                }

                // Add if unique
                if (!already_exists)
                {
                    temp_cols_per_neighbor[neighbor_idx][temp_counts[neighbor_idx]] = col;
                    temp_counts[neighbor_idx]++;
                }
            }
        }
    }

    // Update recv_counts and recv_offsets with actual unique counts
    for (int i = 0; i < n_neighbors; i++)
    {
        halo->recv_counts[i] = temp_counts[i];
    }

    halo->recv_offsets[0] = 0;
    for (int i = 0; i < n_neighbors; i++)
    {
        halo->recv_offsets[i + 1] = halo->recv_offsets[i] + halo->recv_counts[i];
    }
    halo->total_ghost_size = halo->recv_offsets[n_neighbors];

    // Allocate recv_indices with actual size needed
    halo->recv_indices = (int *)malloc(halo->total_ghost_size * sizeof(int));
    if (!halo->recv_indices)
    {
        fprintf(stderr, "[rank %d] Error allocating recv_indices\n", rank);
        for (int i = 0; i < n_neighbors; i++)
        {
            free(temp_cols_per_neighbor[i]);
        }
        free(temp_cols_per_neighbor);
        free(temp_counts);
        free(cols_needed_from_process);
        halo_exchange_free(halo);
        return -1;
    }

    // Copy unique columns to recv_indices
    for (int i = 0; i < n_neighbors; i++)
    {
        int offset = halo->recv_offsets[i];
        for (int j = 0; j < halo->recv_counts[i]; j++)
        {
            halo->recv_indices[offset + j] = temp_cols_per_neighbor[i][j];
        }
    }

    // Free temporary storage
    for (int i = 0; i < n_neighbors; i++)
    {
        free(temp_cols_per_neighbor[i]);
    }
    free(temp_cols_per_neighbor);
    free(temp_counts);
    free(cols_needed_from_process);

    // Step 5: Allocate ghost buffers and build ghost_to_global mapping

    // Allocate ghost_values buffer
    halo->ghost_values = (double *)calloc(halo->total_ghost_size, sizeof(double));
    if (!halo->ghost_values)
    {
        fprintf(stderr, "[rank %d] Error allocating ghost_values\n", rank);
        halo_exchange_free(halo);
        return -1;
    }

    // Allocate and fill ghost_to_global mapping
    halo->ghost_to_global = (int *)malloc(halo->total_ghost_size * sizeof(int));
    if (!halo->ghost_to_global)
    {
        fprintf(stderr, "[rank %d] Error allocating ghost_to_global\n", rank);
        halo_exchange_free(halo);
        return -1;
    }

    // Copy recv_indices to ghost_to_global (they are the same - global column indices)
    for (int i = 0; i < halo->total_ghost_size; i++)
    {
        halo->ghost_to_global[i] = halo->recv_indices[i];
    }

    // Step 6: Communicate with neighbors to determine send_counts and send_indices

    // Allocate send_counts
    halo->send_counts = (int *)calloc(n_neighbors, sizeof(int));
    if (!halo->send_counts)
    {
        fprintf(stderr, "[rank %d] Error allocating send_counts\n", rank);
        halo_exchange_free(halo);
        return -1;
    }

    // Exchange counts:  tell each neighbor how many values we need from them
    MPI_Request *send_requests = (MPI_Request *)malloc(n_neighbors * sizeof(MPI_Request));
    MPI_Request *recv_requests = (MPI_Request *)malloc(n_neighbors * sizeof(MPI_Request));

    if (!send_requests || !recv_requests)
    {
        fprintf(stderr, "[rank %d] Error allocating MPI requests\n", rank);
        free(send_requests);
        free(recv_requests);
        halo_exchange_free(halo);
        return -1;
    }

    // Post receives for send_counts (what others need from us)
    for (int i = 0; i < n_neighbors; i++)
    {
        MPI_Irecv(&halo->send_counts[i], 1, MPI_INT,
                  halo->neighbor_ranks[i], 0, comm, &recv_requests[i]);
    }

    // Send our recv_counts (what we need from them)
    for (int i = 0; i < n_neighbors; i++)
    {
        MPI_Isend(&halo->recv_counts[i], 1, MPI_INT,
                  halo->neighbor_ranks[i], 0, comm, &send_requests[i]);
    }

    // Wait for all count exchanges to complete
    MPI_Waitall(n_neighbors, send_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(n_neighbors, recv_requests, MPI_STATUSES_IGNORE);

    // Build send_offsets
    halo->send_offsets = (int *)malloc((n_neighbors + 1) * sizeof(int));
    if (!halo->send_offsets)
    {
        fprintf(stderr, "[rank %d] Error allocating send_offsets\n", rank);
        free(send_requests);
        free(recv_requests);
        halo_exchange_free(halo);
        return -1;
    }

    halo->send_offsets[0] = 0;
    for (int i = 0; i < n_neighbors; i++)
    {
        halo->send_offsets[i + 1] = halo->send_offsets[i] + halo->send_counts[i];
    }
    int total_send_size = halo->send_offsets[n_neighbors];

    // Allocate send_indices
    halo->send_indices = (int *)malloc(total_send_size * sizeof(int));
    if (!halo->send_indices)
    {
        fprintf(stderr, "[rank %d] Error allocating send_indices\n", rank);
        free(send_requests);
        free(recv_requests);
        halo_exchange_free(halo);
        return -1;
    }

    // Exchange indices: tell neighbors which columns we need, receive which columns they need from us
    for (int i = 0; i < n_neighbors; i++)
    {
        int send_offset = halo->send_offsets[i];
        int send_count = halo->send_counts[i];

        MPI_Irecv(&halo->send_indices[send_offset], send_count, MPI_INT,
                  halo->neighbor_ranks[i], 1, comm, &recv_requests[i]);
    }

    for (int i = 0; i < n_neighbors; i++)
    {
        int recv_offset = halo->recv_offsets[i];
        int recv_count = halo->recv_counts[i];

        MPI_Isend(&halo->recv_indices[recv_offset], recv_count, MPI_INT,
                  halo->neighbor_ranks[i], 1, comm, &send_requests[i]);
    }

    // Wait for all index exchanges to complete
    MPI_Waitall(n_neighbors, send_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(n_neighbors, recv_requests, MPI_STATUSES_IGNORE);

    free(send_requests);
    free(recv_requests);

    // Step 7: Convert send_indices from global to local indexing

    // The send_indices we received are global column indices
    // We need to convert them to local indices in our p_local vector
    for (int i = 0; i < total_send_size; i++)
    {
        int global_col = halo->send_indices[i];

        // Convert global column to local index
        // Our local rows are [row_start, row_end)
        if (global_col >= row_start && global_col < row_end)
        {
            halo->send_indices[i] = global_col - row_start; // Local index
        }
        else
        {
            // This should never happen - neighbor shouldn't request columns we don't own
            fprintf(stderr, "[rank %d] ERROR:  Neighbor requested column %d which is outside our range [%d, %d)\n",
                    rank, global_col, row_start, row_end);
            halo_exchange_free(halo);
            return -1;
        }
    }

    // Success!  Halo exchange structure is fully built
    if (rank == 0)
    {
        printf("[rank %d] Halo exchange ready:  %d total processes using halo communication\n", rank, size);
    }

    return 0;
}

// Add this function after halo_exchange_build

int halo_exchange_execute(HaloExchange *halo, const double *p_local, MPI_Comm comm)
{
    /***
     * Performs halo/ghost exchange.
     * - Sends local values from p_local to neighbors
     * - Receives ghost values and stores in halo->ghost_values
     *
     * Parameters:
     *      halo:        pointer to HaloExchange structure
     *      p_local:    local portion of vector [n_local]
     *      comm:        MPI communicator
     * Returns:  0 on success, -1 on failure
     ***/

    if (halo == NULL || p_local == NULL)
    {
        fprintf(stderr, "ERROR: NULL pointer in halo_exchange_execute\n");
        return -1;
    }

    if (halo->n_neighbors == 0)
    {
        // No communication needed
        return 0;
    }

    // Allocate MPI request arrays
    MPI_Request *send_requests = (MPI_Request *)malloc(halo->n_neighbors * sizeof(MPI_Request));
    MPI_Request *recv_requests = (MPI_Request *)malloc(halo->n_neighbors * sizeof(MPI_Request));

    if (!send_requests || !recv_requests)
    {
        fprintf(stderr, "Error allocating MPI requests in halo_exchange_execute\n");
        free(send_requests);
        free(recv_requests);
        return -1;
    }

    // Allocate temporary send buffer
    int total_send_size = halo->send_offsets[halo->n_neighbors];
    double *send_buffer = (double *)malloc(total_send_size * sizeof(double));

    if (!send_buffer)
    {
        fprintf(stderr, "Error allocating send buffer in halo_exchange_execute\n");
        free(send_requests);
        free(recv_requests);
        return -1;
    }

    // Pack send buffer:  gather local values that neighbors need
    for (int i = 0; i < total_send_size; i++)
    {
        int local_idx = halo->send_indices[i];
        send_buffer[i] = p_local[local_idx];
    }

    // Post non-blocking receives for ghost values
    for (int i = 0; i < halo->n_neighbors; i++)
    {
        int recv_offset = halo->recv_offsets[i];
        int recv_count = halo->recv_counts[i];

        MPI_Irecv(&halo->ghost_values[recv_offset], recv_count, MPI_DOUBLE,
                  halo->neighbor_ranks[i], 2, comm, &recv_requests[i]);
    }

    // Post non-blocking sends for local values
    for (int i = 0; i < halo->n_neighbors; i++)
    {
        int send_offset = halo->send_offsets[i];
        int send_count = halo->send_counts[i];

        MPI_Isend(&send_buffer[send_offset], send_count, MPI_DOUBLE,
                  halo->neighbor_ranks[i], 2, comm, &send_requests[i]);
    }

    // Wait for all communication to complete
    MPI_Waitall(halo->n_neighbors, recv_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(halo->n_neighbors, send_requests, MPI_STATUSES_IGNORE);

    // Clean up
    free(send_buffer);
    free(send_requests);
    free(recv_requests);

    return 0;
}