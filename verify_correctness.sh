#!/bin/bash
#PBS -N verify_CG_correctness
#PBS -l select=4:ncpus=96:mpiprocs=96:mem=192gb
#PBS -l place=scatter:excl
#PBS -l walltime=01:00:00
#PBS -q short_HPC4DS
#PBS -j oe

cd $PBS_O_WORKDIR
module load mpich-3.2

# Define problem size
GRID=512

echo "Running Verification on 4 Nodes (384 Ranks)..."

# 1. Run Baseline (Ensure it prints residual every 5-10 iterations to logs)
mpirun.actual -n 384 --bind-to core ./build/solver_baseline $GRID > logs/verify_baseline.log 2>&1

# 2. Run Pipelined
mpirun.actual -n 384 --bind-to core ./build/solver_pipelined $GRID > logs/verify_pipelined.log 2>&1

echo "Verification Jobs Completed."