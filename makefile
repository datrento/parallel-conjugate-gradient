CC = mpicc
CSTD = -std=c99
OPTFLAGS = -O3 -march=native -ffast-math -funroll-loops -fno-math-errno
COMMON_FLAGS = $(CSTD) -Wall -Wextra
CFLAGS = $(COMMON_FLAGS) $(OPTFLAGS)
LDFLAGS = -lm
INC = -Iinclude

# Directories
SRC_DIR = src
BUILD_DIR = build
OUTPUT_DIR = output
LOGS_DIR = logs

# Source files
SOURCES = \
	$(SRC_DIR)/main_parallel_csr.c \
	$(SRC_DIR)/csr.c \
	$(SRC_DIR)/solver.c \
	$(SRC_DIR)/solver_utils.c \
	$(SRC_DIR)/utils.c \
	$(SRC_DIR)/csr_matrix_builder.c

# Build options
debug ?= 0
omp ?= 0

DEFS :=

ifeq ($(debug),1)
	DEFS += -DDEBUG
	CFLAGS := $(COMMON_FLAGS) -O0 -g
endif
ifeq ($(omp),1)
	CFLAGS += -fopenmp
	LDFLAGS += -fopenmp
endif

# Dual Targets
TARGET_BASE = $(BUILD_DIR)/solver_baseline
TARGET_PIPE = $(BUILD_DIR)/solver_pipelined

all: compile print_config

dirs:
	mkdir -p $(BUILD_DIR) $(LOGS_DIR) $(OUTPUT_DIR)

# Build Baseline Binary (Force BASE_LINE_ONLY=1)
compile_baseline: dirs $(SOURCES)
	@echo "Compiling Baseline CG Solver..."
	$(CC) $(CFLAGS) $(INC) $(DEFS) -DBASE_LINE_ONLY=1 -o $(TARGET_BASE) $(SOURCES) $(LDFLAGS)
	@echo "Baseline solver compiled at $(TARGET_BASE)"

compile_pipelined: dirs $(SOURCES)
	@echo "Compiling Pipelined CG Solver..."
	$(CC) $(CFLAGS) $(INC) $(DEFS) -DBASE_LINE_ONLY=0 -o $(TARGET_PIPE) $(SOURCES) $(LDFLAGS)
	@echo "Pipelined solver compiled at $(TARGET_PIPE)"

# Compile both versions so the benchmarking script can use both
compile: dirs compile_baseline compile_pipelined
	@echo "Compilation finished. Binaries at $(TARGET_BASE) and $(TARGET_PIPE)"
	
print_config:
	@echo "=== Solver Configuration ==="
	@echo "Compiler: $(CC)"
	@echo "Flags:    $(CFLAGS)"
	@echo "Reps:     $(REPS)"
	@echo "Binary Base: $(TARGET_BASE)"
	@echo "Binary Pipe: $(TARGET_PIPE)"
	@echo "============================"
		sleep 1; \
	done
	@echo "All $(REPS) repetitions submitted."

# disable this in the cluster environment
local_run: compile
	mpirun -n 8 $(TARGET_PIPE) 200 > $(LOGS_DIR)/local_run_out_pipe.log 2> $(LOGS_DIR)/local_run_err_pipe.log
	mpirun -n 8 $(TARGET_BASE) 200 >> $(LOGS_DIR)/local_run_out_base.log 2>> $(LOGS_DIR)/local_run_err_base.log
	@echo "Local run completed. Output in $(LOGS_DIR)/local_run_out.log, errors in $(LOGS_DIR)/local_run_err.log"

clean:
	rm -rf $(BUILD_DIR) $(LOGS_DIR)/* $(OUTPUT_DIR)/*
	@echo "Cleaned build, logs, and output."

.PHONY: all compile dirs clean local_run compile_baseline compile_pipelined 