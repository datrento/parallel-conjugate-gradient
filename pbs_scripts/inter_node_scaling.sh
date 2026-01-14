#!/bin/bash
#PBS -N CG_InterNode_StrongScaling
#PBS -l select=4:ncpus=96:mpiprocs=96:mem=800gb
#PBS -l place=scatter:excl
#PBS -l walltime=06:00:00
#PBS -q short_HPC4DS
#PBS -j oe

cd $PBS_O_WORKDIR

# Create a UNIQUE workspace for Inter-node data
WORK_DIR="workspace_inter_node"
mkdir -p $WORK_DIR/output
mkdir -p $WORK_DIR/logs

# Copy the binaries into this workspace
cp build/solver_baseline $WORK_DIR/
cp build/solver_pipelined $WORK_DIR/

cd $WORK_DIR

# Load the environment
module load mpich-3.2

# Define the grid sizes (N values)
# N=512 (~2 min serial), N=875 (~10 min), N=1100 (~20 min)
GRIDS=(512 875 1100)

# Define the node counts for scaling
NODE_COUNTS=(1 2 4)

# Parameters
MAX_ITER=1000
TOL=1e-10

echo "--- EXPERIMENT: MULTI-NODE STRONG SCALING ---"
echo "Start Time: $(date)"
echo "Hardware Check: Xeon 6252N (96 cores/node)"

for GRID in "${GRIDS[@]}"; do
    echo "=========================================================="
    echo "  GRID SIZE: $GRID x $GRID x $GRID"
    echo "=========================================================="

    for NODES in "${NODE_COUNTS[@]}"; do
        RANKS=$((NODES * 96))
        echo ">>> Testing $NODES Node(s) ($RANKS Ranks)..."

        # 1. BASELINE CG - 3 Replications
        for i in {1..3}; do
            echo "Baseline Repetition $i..."
            # Using --bind-to core and --map-by socket for performance
            mpirun.actual -n $RANKS --bind-to core --map-by socket ./build/solver_baseline $GRID $MAX_ITER $TOL > logs/baseline_${GRID}grid_${RANKS}ranks_rep${i}.log 2>&1
        done

        # 2. PIPELINED CG - 3 Replications
        for i in {1..3}; do
            echo "Pipelined Repetition $i..."
            mpirun.actual -n $RANKS --bind-to core --map-by socket ./build/solver_pipelined $GRID $MAX_ITER $TOL > logs/pipelined_${GRID}grid_${RANKS}ranks_rep${i}.log 2>&1
        done
        
        echo "----------------------------------------------------------"
    done
done

echo "End Time: $(date)"
echo "Benchmark Complete. Check output/jcgtimes.txt for data."