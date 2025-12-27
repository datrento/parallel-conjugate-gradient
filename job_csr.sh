#! /bin/bash

# set job name
#PBS -N conjugate_gradient_job

# set number of chunks, processors per node, and memory
#PBS -l select=2:ncpus=2:mem=128gb -l place=scatter:excl

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
grid_sizes=(480 850 1200)

# Loop over each grid size
for grid in "${grid_sizes[@]}"; do
    echo "Running grid_size=$grid"

    # Redirect stdout and stderr to separate files per grid size
    # usage: mpirun.actual -n <num_processes> ./build/main_parallel_csr <grid_size> <max_iter> <tolerance>
    mpirun.actual -n 2 ./build/main_parallel_csr $grid \
        > logs/output_grid_${grid}.log \
        2> logs/error_grid_${grid}.log 
done

