#!/bin/bash
#PBS -N CG_Strong_875_1100
#PBS -l select=4:ncpus=96:mpiprocs=96:mem=800gb
#PBS -l place=scatter:excl
#PBS -l walltime=06:00:00
#PBS -q short_HPC4DS
#PBS -j oe

cd $PBS_O_WORKDIR

# 1. Unique Workspace
WORK_DIR="run_n875_1100_strong_$PBS_JOBID"
mkdir -p $WORK_DIR/output
mkdir -p $WORK_DIR/logs
echo "Working directory: $WORK_DIR"

# 2. Binaries
cp build/solver_baseline $WORK_DIR/
cp build/solver_pipelined $WORK_DIR/
cd $WORK_DIR

# Load Environment and Verify Hardware
module load mpich-3.2
echo "--- Hardware Info for N=875 ---"
hostname
lscpu | grep "Model name"
echo "-------------------------------"

# 4. Run

for GRID in 875 1100; do
    for NODES in 1 2 4; do
        RANKS=$((NODES * 96))
        for i in {1..3}; do
            # 1. PIPELINED: Run 1000 iterations (Fast)
            mpirun.actual -n $RANKS --bind-to core --map-by socket ./solver_pipelined $GRID 1000 1e-10 > logs/pipelined_${GRID}_${RANKS}_rep${i}.log 2>&1
            
            # 2. BASELINE PROBE: Run 20 iterations (Manageable)
            # If N=1100 Baseline crashes, you only lose a few seconds
            mpirun.actual -n $RANKS --bind-to core --map-by socket ./solver_baseline $GRID 20 1e-10 > logs/baseline_${GRID}_${RANKS}_probe_rep${i}.log 2>&1
        done
    done
done