#!/bin/bash
#PBS -N CG_Weak_Scaling
#PBS -l select=4:ncpus=96:mpiprocs=96:mem=800gb
#PBS -l place=scatter:excl
#PBS -l walltime=06:00:00
#PBS -q short_HPC4DS
#PBS -j oe

cd $PBS_O_WORKDIR

WORK_DIR="weak_scaling_$PBS_JOBID"
mkdir -p $WORK_DIR/output
mkdir -p $WORK_DIR/logs
cp build/solver_baseline $WORK_DIR/
cp build/solver_pipelined $WORK_DIR/
cd $WORK_DIR

module load mpich-3.2

# --- EXPERIMENT 1: 2 NODES (192 RANKS), N=645 ---
# Work/Rank is identical to 512 grid on 96 ranks
for i in {1..3}; do
    echo "Starting N=645 (192 Ranks) - Repetition $i"
    # Pipelined (Full run for convergence - expected ~50-60s)
    mpirun.actual -n 192 ./solver_pipelined 645 1000 1e-10 > logs/pipelined_645_192_rep${i}.log 2>&1
    
    # Baseline (Probe - 20 iterations is enough to get avg_iter_time)
    # This prevents the 6-hour walltime limit from killing the job
    mpirun.actual -n 192 ./solver_baseline 645 20 1e-10 > logs/baseline_645_192_probe_rep${i}.log 2>&1
done

# --- EXPERIMENT 2: 4 NODES (384 RANKS), N=813 ---
# Work/Rank is identical to 512 grid on 96 ranks
for i in {1..3}; do
    echo "Starting N=813 (384 Ranks) - Repetition $i"
    # Pipelined (Full run - expected ~50-60s)
    mpirun.actual -n 384 ./solver_pipelined 813 1000 1e-10 > logs/pipelined_813_384_rep${i}.log 2>&1
    
    # Baseline (Probe - 20 iterations)
    mpirun.actual -n 384 ./solver_baseline 813 20 1e-10 > logs/baseline_813_384_probe_rep${i}.log 2>&1
done

echo "Weak Scaling Experiments Complete."