#!/bin/bash
#PBS -N CG_IntraNode_ScaleUp
#PBS -l select=1:ncpus=96:mpiprocs=96:mem=400gb
#PBS -l place=scatter:excl
#PBS -l walltime=06:00:00
#PBS -q short_HPC4DS
#PBS -j oe

cd $PBS_O_WORKDIR
module load mpich-3.2

GRID=875
CORE_COUNTS=(1 2 4 8 16 32 48 64 80 96)

echo "--- Experiment: Intra-Node Scale-Up (Baseline vs Pipelined) ---"

for CORES in "${CORE_COUNTS[@]}"; do
    echo "================================"
    echo "CORES: $CORES"
    echo "================================"
    
    # 1. Run Baseline 3 times
    for i in {1..3}; do
        echo "Baseline Rep $i..."
        mpirun.actual -n $CORES --bind-to core --map-by socket ./build/solver_baseline $GRID 1000 1e-10
    done

    # 2. Run Pipelined 3 times
    for i in {1..3}; do
        echo "Pipelined Rep $i..."
        mpirun.actual -n $CORES --bind-to core --map-by socket ./build/solver_pipelined $GRID 1000 1e-10
    done
done