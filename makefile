CC = mpicc
CFLAGS = -std=c99 -O3 -Wall
LDFLAGS = -lm

# Directories
SRC_DIR = src
BUILD_DIR = build
LOGS_DIR = logs

# Files
#Executable target
TARGET = $(BUILD_DIR)/main_parallel_csr

# Source files
SOURCES = $(SRC_DIR)/main_parallel_csr.c $(SRC_DIR)/utils.c $(SRC_DIR)/csr.c

# PBS Script in root
PBS_SCRIPT = job_csr.sh


all: compile print_config

dirs:
	mkdir -p $(BUILD_DIR) $(LOGS_DIR)
	@echo "Created directories: $(BUILD_DIR), $(LOGS_DIR)"

# Compile program into build/
compile: dirs
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)
	@echo "Compilation finished. Executable at $(TARGET)"

# Submit job to the queue
submit: compile print_config
	qsub $(PBS_SCRIPT) | tee $(LOGS_DIR)/job_id.txt
	@echo "Job submitted. Job ID stored in $(LOGS_DIR)/job_id.txt"

print_config:
	@echo "Makefile Configuration:"
	@echo "Compiler: $(CC)"
	@echo "Source directory: $(SRC_DIR)"
	@echo "Build directory: $(BUILD_DIR)"
	@echo "Logs directory: $(LOGS_DIR)"
	@echo "Executable target: $(TARGET)"
status:
	qstat $$(cat $(LOGS_DIR)/job_id.txt)

my_jobs:
	qstat -u $$USER

# Run locally with 4 processes and grid size 100
# usage: make local_run <grid_size> <max_iter> <tolerance>

# disable this in the cluster environment
local_run: compile
	mpirun -n 4 $(TARGET) 300 > $(LOGS_DIR)/local_run_out.log 2> $(LOGS_DIR)/local_run_err.log
	@echo "Local run completed. Output in $(LOGS_DIR)/local_run_out.log, errors in $(LOGS_DIR)/local_run_err.log"

# Clean everything generated
clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(LOGS_DIR)/*.log *.out *.err
	@echo "Cleaned build and log files."

.PHONY: all compile submit clean dirs status my_jobs local_run print_config
