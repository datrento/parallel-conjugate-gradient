#!/bin/bash
#PBS -N CG_Weak_Scaling
#PBS -l select=8:ncpus=96:mpiprocs=96:mem=400gb
#PBS -l place=scatter:excl
#PBS -q short_HPC4DS
#PBS -j oe

cd $PBS_O_WORKDIR
module load mpich-3.2

# Load per rank: ~1.5 Million unknowns
# 1 Node (96 ranks): Grid 524  (~144M total)
# 2 Nodes (192 ranks): Grid 660 (~288M total)
# 4 Nodes (384 ranks): Grid 832 (~576M total)
# 8 Nodes (768 ranks): Grid 1048 (~1.1B total)

declare -A GRIDS=( [96]=524 [192]=660 [384]=832 [768]=1048 )

for NODES in 1 2 4 8; do
    RANKS=$((NODES * 96))
    G=${GRIDS[$RANKS]}
    
    echo "--- Testing Weak Scaling: $NODES nodes, Grid $G ---"
    mpirun.actual -n $RANKS --bind-to core ./build/solver_baseline $G 1000 1e-10
    mpirun.actual -n $RANKS --bind-to core ./build/solver_pipelined $G 1000 1e-10
done