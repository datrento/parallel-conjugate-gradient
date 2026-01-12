#!/bin/bash
#PBS -N CG_Weak_Scaling_commtime
#PBS -l select=4:ncpus=96:mpiprocs=96:mem=800gb
#PBS -l place=scatter:excl
#PBS -l walltime=06:00:00
#PBS -q short_HPC4DS
#PBS -j oe

# Move to the directory where the job was submitted
cd $PBS_O_WORKDIR

# Create a unique directory for this scaling study
WORK_DIR="scaling_commtime_$PBS_JOBID"
mkdir -p $WORK_DIR/output
mkdir -p $WORK_DIR/logs
cp build/solver_baseline $WORK_DIR/
cp build/solver_pipelined $WORK_DIR/
cd $WORK_DIR

# Load MPI module
module load mpich-3.2

# Configuration: 3 repetitions to ensure statistical stability
REPS=3
TOL="1e-10"

echo "-------------------------------------------------------"
echo "Starting Weak Scaling Experiment"
echo "Work-per-rank constant (relative to N=512 on 96 ranks)"
echo "-------------------------------------------------------"

# --- CASE 1: 1 NODE (96 RANKS), N=512 ---
# (Reference Case)
for i in $(seq 1 $REPS); do
    echo "[N=512, Ranks=96] Repetition $i"
    # Pipelined Full Run
    mpirun.actual -n 96 ./solver_pipelined 512 1000 $TOL > logs/pipe_512_96_rep${i}.log 2>&1
    # Baseline Probe (21 iterations to get average)
    mpirun.actual -n 96 ./solver_baseline 512 1000 $TOL > logs/base_512_96_rep${i}.log 2>&1
done

# --- CASE 2: 2 NODES (192 RANKS), N=645 ---
for i in $(seq 1 $REPS); do
    echo "[N=645, Ranks=192] Repetition $i"
    # Pipelined Full Run
    mpirun.actual -n 192 ./solver_pipelined 645 1000 $TOL > logs/pipe_645_192_rep${i}.log 2>&1
    # Baseline Probe
    mpirun.actual -n 192 ./solver_baseline 645 21 $TOL > logs/base_645_192_rep${i}.log 2>&1
done

# --- CASE 3: 4 NODES (384 RANKS), N=813 ---
for i in $(seq 1 $REPS); do
    echo "[N=813, Ranks=384] Repetition $i"
    # Pipelined Full Run
    mpirun.actual -n 384 ./solver_pipelined 813 1000 $TOL > logs/pipe_813_384_rep${i}.log 2>&1
    # Baseline Probe
    mpirun.actual -n 384 ./solver_baseline 813 21 $TOL > logs/base_813_384_rep${i}.log 2>&1
done

echo "-------------------------------------------------------"
echo "Experiments Completed Successfully."
echo "Results located in: $WORK_DIR/output/jcgtimes.txt"
echo "Detailed logs in: $WORK_DIR/logs/"
echo "-------------------------------------------------------"