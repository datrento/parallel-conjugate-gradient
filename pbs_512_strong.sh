#!/bin/bash
#PBS -N CG_Strong_512
#PBS -l select=4:ncpus=96:mpiprocs=96:mem=800gb
#PBS -l place=scatter:excl
#PBS -l walltime=02:00:00
#PBS -q short_HPC4DS
#PBS -j oe

cd $PBS_O_WORKDIR

# 1. Create UNIQUE workspace for this grid size
WORK_DIR="run_n512_strong_$PBS_JOBID"
mkdir -p $WORK_DIR/output
mkdir -p $WORK_DIR/logs

# 2. Copy binaries into the workspace
cp build/solver_baseline $WORK_DIR/
cp build/solver_pipelined $WORK_DIR/
cd $WORK_DIR

# 3. Load Environment and Verify Hardware
module load mpich-3.2
echo "--- Hardware Info for N=512 ---"
hostname
lscpu | grep "Model name"
echo "-------------------------------"

# 4. Parameters
GRID=512
NODE_COUNTS=(1 2 4)

for NODES in "${NODE_COUNTS[@]}"; do
    RANKS=$((NODES * 96))
    for i in {1..3}; do
        # Baseline
        mpirun.actual -n $RANKS --bind-to core --map-by socket ./solver_baseline $GRID 1000 1e-10 > logs/baseline_${GRID}_${RANKS}_rep${i}.log 2>&1
        # Pipelined
        mpirun.actual -n $RANKS --bind-to core --map-by socket ./solver_pipelined $GRID 1000 1e-10 > logs/pipelined_${GRID}_${RANKS}_rep${i}.log 2>&1
    done
done