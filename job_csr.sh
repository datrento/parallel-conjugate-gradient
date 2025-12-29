#! /bin/bash

# set job name
#PBS -N conjugate_gradient_job
# set number of chunks, processors per node, and memory
#PBS -l select=32:ncpus=1:mem=192gb -l place=scatter:excl

# set maximum walltime
#PBS -l walltime=03:00:00

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
#Define grid sizes and number of processors
grid_sizes=(380 500 640)
nprocs_list=(1 2 4 8 16 32)

for grid in "${grid_sizes[@]}"; do
  for np in "${nprocs_list[@]}"; do
    echo "Running grid_size=$grid with np=$np"
    mpirun.actual -n "$np" ./build/main_parallel_csr "$grid" \
      > "logs/output_grid_${grid}_np_${np}.log" \
      2> "logs/error_grid_${grid}_np_${np}.log"
  done
done

