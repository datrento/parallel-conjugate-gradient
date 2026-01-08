#!/bin/bash
#PBS -N CG_Strong_Scaling
#PBS -l select=8:ncpus=96:mpiprocs=96:mem=200gb
#PBS -l place=scatter:excl
#PBS -q short_HPC4DS
#PBS -j oe

cd $PBS_O_WORKDIR
module load mpich-3.2

# Fixed Grid: 220^3 is ~10.6M unknowns. 
# This is small enough to hit the "wall" at 8 nodes (768 cores).
GRID=220

echo "Node_Count,Ranks,Method,Total_Time,Iter_Time,Iters"
for NODES in 1 2 4 8; do
    RANKS=$((NODES * 96))
    # Run Baseline
    mpirun.actual -n $RANKS --bind-to core ./build/solver_baseline $GRID 1000 1e-10
    # Run Pipelined
    mpirun.actual -n $RANKS --bind-to core ./build/solver_pipelined $GRID 1000 1e-10
done