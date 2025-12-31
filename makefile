CC = mpicc
CSTD = -std=c99
CFLAGS = $(CSTD) -O3 -Wall
LDFLAGS = -lm
INC = -Iinclude

# Directories
SRC_DIR = src
BUILD_DIR = build
OUTPUT_DIR = output
LOGS_DIR = logs

# Files
#Executable target
TARGET = $(BUILD_DIR)/main_parallel_csr

# Source files
SOURCES = \
	$(SRC_DIR)/main_parallel_csr.c \
	$(SRC_DIR)/csr.c \
	$(SRC_DIR)/solver.c \
	$(SRC_DIR)/solver_utils.c \
	$(SRC_DIR)/utils.c \
	$(SRC_DIR)/csr_matrix_builder.c
# $(SRC_DIR)/dense_utils.c but not used in CSR version


# Build toggles
overlap ?= 0
debug ?= 0
omp ?= 0

DEFS :=
ifeq ($(overlap),1)
	DEFS += -DENABLE_OVERLAP
endif
ifeq ($(debug),1)
	DEFS += -DDEBUG
	CFLAGS := $(CSTD) -O0 -g -Wall
endif
ifeq ($(omp),1)
	CFLAGS += -fopenmp
	LDFLAGS += -fopenmp
endif

# PBS Script in root
PBS_SCRIPT = run_single_rep.sh
PBS_SINGLE_REP = ${PBS_SCRIPT}

# Number of repetitions
REPS = 3

all: compile print_config

dirs:
	mkdir -p $(BUILD_DIR) $(LOGS_DIR) $(OUTPUT_DIR)
	@echo "Created directories: $(BUILD_DIR), $(LOGS_DIR) and $(OUTPUT_DIR)"

# Compile program into build/
compile: dirs
	$(CC) $(CFLAGS) $(INC) $(DEFS) -o $(TARGET) $(SOURCES) $(LDFLAGS)
	@echo "Compilation finished. Executable at $(TARGET)"

# Submit job to the queue
submit: compile print_config
	qsub $(PBS_SCRIPT) | tee $(LOGS_DIR)/job_id.txt
	@echo "Job submitted. Job ID stored in $(LOGS_DIR)/job_id.txt"

# Submit multiple independent runs for better statistics
submit_reps: compile print_config
	@echo "Submitting $(REPS) independent repetitions..."
	@for rep in $$(seq 1 $(REPS)); do \
		echo "Submitting repetition $$rep... "; \
		qsub -v REP=$$rep $(PBS_SINGLE_REP) | tee $(LOGS_DIR)/job_id_rep$$rep.txt; \
		sleep 1; \
	done
	@echo "All $(REPS) jobs submitted. Check status with: make status_reps"

print_config:
	@echo "Makefile Configuration:"
	@echo "Compiler: $(CC)"
	@echo "Source directory: $(SRC_DIR)"
	@echo "Build directory: $(BUILD_DIR)"
	@echo "Logs directory: $(LOGS_DIR)"
	@echo "Executable target: $(TARGET)"
	@echo "CFLAGS: $(CFLAGS) $(DEFS)"
	@echo "LDFLAGS: $(LDFLAGS)"
# Trace job information
trace:
	@echo "Tracing main job:"
	@if [ -f $(LOGS_DIR)/job_id.txt ]; then \
		tracejob $$(cat $(LOGS_DIR)/job_id.txt); \
	else \
		echo "No job_id.txt found. "; \
	fi

# Trace all repetition jobs
trace_reps:
	@echo "Tracing repetition jobs:"
	@for rep in $$(seq 1 $(REPS)); do \
		if [ -f $(LOGS_DIR)/job_id_rep$$rep.txt ]; then \
			job_id=$$(cat $(LOGS_DIR)/job_id_rep$$rep.txt); \
			echo "=== Repetition $$rep (Job ID: $$job_id) ==="; \
			tracejob $$job_id 2>/dev/null || echo "No trace available"; \
		fi; \
	done

status:
	@echo "Status of main job:"
	@if [ -f $(LOGS_DIR)/job_id.txt ]; then \
		qstat $$(cat $(LOGS_DIR)/job_id.txt) -H; \
	else \
		echo "No job_id.txt found."; \
	fi

# Check status of all repetition jobs
status_reps: 
	@echo "Status of repetition jobs:"
	@for rep in $$(seq 1 $(REPS)); do \
		if [ -f $(LOGS_DIR)/job_id_rep$$rep.txt ]; then \
			job_id=$$(cat $(LOGS_DIR)/job_id_rep$$rep.txt); \
			echo "=== Repetition $$rep (Job ID: $$job_id) ==="; \
			qstat $$job_id 2>/dev/null || echo "Job completed or not found"; \
		fi; \
	done

my_jobs:
	qstat -u $$USER

# Run locally with 4 processes and grid size 100
# usage: make local_run <grid_size> <max_iter> <tolerance>

# disable this in the cluster environment
local_run: compile
	mpirun -n 4 $(TARGET) 10 > $(LOGS_DIR)/local_run_out.log 2> $(LOGS_DIR)/local_run_err.log
	@echo "Local run completed. Output in $(LOGS_DIR)/local_run_out.log, errors in $(LOGS_DIR)/local_run_err.log"

# Clean everything generated
clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(LOGS_DIR)/*.log
	rm -rf $(LOGS_DIR)/job_id.txt
	rm -rf $(OUTPUT_DIR)/*.txt
	@echo "Cleaned build and log files."

# Clean only repetition logs
clean_reps:
	rm -rf $(LOGS_DIR)/rep*
	rm -rf $(LOGS_DIR)/job_id_rep*.txt
	rm -rf $(LOGS_DIR)/*_rep*.log
	@echo "Cleaned repetition logs."

.PHONY: all compile submit submit_reps clean clean_reps dirs status status_reps my_jobs local_run print_config trace trace_reps