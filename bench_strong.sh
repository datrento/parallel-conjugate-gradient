#!/bin/bash
#PBS -N CG_Strong_Scaling_27pt
#PBS -l select=4:ncpus=96:mpiprocs=96:mem=300gb
#PBS -l place=scatter:excl
#PBS -l walltime=02:00:00
#PBS -q short_HPC4DS
#PBS -j oe

cd $PBS_O_WORKDIR
module load mpich-3.2

# Fixed Problem Size (134M unknowns)
GRID=512
# Iterating through node counts: 1, 2, 4 nodes
NODE_COUNTS=(1 2 4)

echo "--- Experiment: Strong Scaling (Grid=$GRID) ---"

for NODES in "${NODE_COUNTS[@]}"; do
    RANKS=$((NODES * 96))
    echo "Testing $NODES Nodes ($RANKS Ranks)..."

    # Run Baseline 3 times
    for i in {1..3}; do
        echo "Baseline Rep $i..."
        mpirun.actual -n $RANKS --bind-to core ./build/solver_baseline $GRID 1000 1e-10
    done

    # Run Pipelined 3 times
    for i in {1..3}; do
        echo "Pipelined Rep $i..."
        mpirun.actual -n $RANKS --bind-to core ./build/solver_pipelined $GRID 1000 1e-10
    done
done