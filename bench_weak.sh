#!/bin/bash
#PBS -N CG_Weak_Scaling_27pt
#PBS -l select=4:ncpus=96:mpiprocs=96:mem=600gb
#PBS -l place=scatter:excl
#PBS -l walltime=02:00:00
#PBS -q short_HPC4DS
#PBS -j oe

cd $PBS_O_WORKDIR
module load mpich-3.2

# Pre-calculated Grid Sizes for 1.5M points per rank:
# 96 ranks:  N=524
# 192 ranks: N=660
# 384 ranks: N=832

declare -A WEAK_GRIDS=( [96]=524 [192]=660 [384]=832 )
NODE_COUNTS=(1 2 4)

echo "--- Experiment: Weak Scaling (1.5M points/rank) ---"

for NODES in "${NODE_COUNTS[@]}"; do
    RANKS=$((NODES * 96))
    G=${WEAK_GRIDS[$RANKS]}
    echo "Testing $NODES Nodes ($RANKS Ranks) with Grid $G..."

    # Run Baseline 3 times
    for i in {1..3}; do
        mpirun.actual -n $RANKS --bind-to core ./build/solver_baseline $G 1000 1e-10
    done

    # Run Pipelined 3 times
    for i in {1..3}; do
        mpirun.actual -n $RANKS --bind-to core ./build/solver_pipelined $G 1000 1e-10
    done
done