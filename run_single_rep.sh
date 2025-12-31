#!/bin/bash
#PBS -N conjugate_gradient_job
#PBS -l select=32:ncpus=1:mem=192gb -l place=scatter:excl
#PBS -l walltime=04:00:00
#PBS -q short_HPC4DS
#PBS -j oe

cd $PBS_O_WORKDIR
module load mpich-3.2

echo "=========================================="
if [ -z "$REP" ] || [ "$REP" -eq 0 ]; then
  echo "Starting Single Run"
  REP=0
else
  echo "Starting Repetition ${REP}"
fi
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "=========================================="

grid_sizes=(200 380 500 640)
nprocs_list=(1 2 4 8 16 32)

for grid in "${grid_sizes[@]}"; do
  for np in "${nprocs_list[@]}"; do
    
    # Determine output file names based on REP value
    if [ "$REP" -eq 0 ]; then
      # Single run - no rep prefix
      output_log="logs/grid${grid}_np${np}.log"
      error_log="logs/grid${grid}_np${np}.err"
      echo "Running grid=$grid np=$np at $(date +%H:%M:%S)"
    else
      # Multiple reps - include rep prefix
      output_log="logs/rep${REP}_grid${grid}_np${np}. log"
      error_log="logs/rep${REP}_grid${grid}_np${np}.err"
      echo "[Rep ${REP}] Running grid=$grid np=$np at $(date +%H:%M:%S)"
    fi
    
    # Run the program
    mpirun.actual -n "$np" ./build/main_parallel_csr "$grid" \
      > "$output_log" \
      2> "$error_log"
    
    # Capture exit status
    status=$? 
    if [ $status -ne 0 ]; then
      echo "ERROR: grid=$grid np=$np failed with status $status"
    fi
  done
done

echo "=========================================="
if [ "$REP" -eq 0 ]; then
  echo "Completed Single Run at $(date)"
else
  echo "Completed Repetition ${REP} at $(date)"
fi
echo "=========================================="