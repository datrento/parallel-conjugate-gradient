#! /bin/bash

# set job name
#PBS -N conjugate_gradient_job

# set number of chunks, processors per node, and memory
#PBS -l select=1:ncpus=1:mem=128gb

# set maximum walltime
#PBS -l walltime=00:20:00

# set queue
#PBS -q short_cpuQ

# set output and error files
#PBS -o output.log
#PBS -e error.log

# change to the directory from which the job was submitted
cd $PBS_O_WORKDIR

# load necessary modules
module load mpich-3.2

# run the parallel program
# mpirun.actual -n 32 ./src/main_parallel_csr

# Array of grid sizes for small, medium, large runs
grid_sizes=(300 600 900)

# Loop over each grid size
for grid in "${grid_sizes[@]}"; do
    echo "Running grid_size=$grid"

    # Redirect stdout and stderr to separate files per grid size
    mpirun.actual -n 32 ./src/main_parallel_csr $grid \
        > output_grid_${grid}.log \
        2> error_grid_${grid}.log 
done

