# Compilers — do not assume nvcc lives under CONDA_PREFIX (often only in CUDA toolkit).
# Override: make NVCC=/usr/local/cuda/bin/nvcc
# Priority: $CUDA_HOME/bin, /usr/local/cuda/bin, conda prefix if nvcc exists there, else nvcc on PATH.
ifndef NVCC
  NVCC :=
  ifneq ($(strip $(CUDA_HOME)),)
    ifneq ($(wildcard $(CUDA_HOME)/bin/nvcc),)
      NVCC := $(CUDA_HOME)/bin/nvcc
    endif
  endif
  ifeq ($(NVCC),)
    ifneq ($(wildcard /usr/local/cuda/bin/nvcc),)
      NVCC := /usr/local/cuda/bin/nvcc
    endif
  endif
  ifeq ($(NVCC),)
    ifneq ($(strip $(CONDA_PREFIX)),)
      ifneq ($(wildcard $(CONDA_PREFIX)/bin/nvcc),)
        NVCC := $(CONDA_PREFIX)/bin/nvcc
      endif
    endif
  endif
  ifeq ($(NVCC),)
    NVCC := nvcc
  endif
endif

# Include and library paths
INCLUDES ?= $(if $(CONDA_PREFIX),-I$(CONDA_PREFIX)/include,)
LIBS ?= $(if $(CONDA_PREFIX),-L$(CONDA_PREFIX)/lib,)

# Host compiler for nvcc (CUDA 12.1 typically supports GCC <= 12 on Linux).
# Override via: make HOST_COMPILER=g++-12
HOST_COMPILER ?=
ifeq ($(strip $(HOST_COMPILER)),)
  # If user didn't specify, prefer g++-12 when available.
  HOST_COMPILER := $(shell command -v g++-12 2>/dev/null)
endif
ifneq ($(strip $(HOST_COMPILER)),)
  NVCC_HOST_FLAGS := -ccbin=$(HOST_COMPILER)
else
  NVCC_HOST_FLAGS :=
endif

# GPU architecture (override: make SM=90).
# Default: GPU 0 compute capability from nvidia-smi (e.g. 9.0 -> 90, 8.9 -> 89). Falls back to 86.
SM ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.')
ifeq ($(strip $(SM)),)
  SM := 86
endif
# If SM or nvcc target changed, rebuild binaries: make clean && make rmsnorm (or make all).

# NVCC compiler flags
# Build both native SASS and PTX for forward compatibility.
# PTX lets newer GPUs JIT-compile even if their exact SM isn't supported by this CUDA version.
NVCC_GENCODE ?= -gencode arch=compute_$(SM),code=sm_$(SM) -gencode arch=compute_$(SM),code=compute_$(SM)
NVCC_ALLOW_UNSUPPORTED ?= 0
ifeq ($(NVCC_ALLOW_UNSUPPORTED),1)
  NVCC_UNSUPPORTED_FLAGS := -allow-unsupported-compiler
else
  NVCC_UNSUPPORTED_FLAGS :=
endif

NVCC_FLAGS := $(NVCC_GENCODE) $(NVCC_HOST_FLAGS) $(NVCC_UNSUPPORTED_FLAGS) -std=c++17

# Device link-time optimization (LTO)
# Some nvcc versions reject mixing -dlto with explicit -gencode. Keep it off by default.
USE_LTO ?= 0
ifeq ($(USE_LTO),1)
  NVCC_LTO_FLAGS := -dlto
else
  NVCC_LTO_FLAGS :=
endif

# Source and object files
CU_SRCS := bls12-381.cu ioutils.cu commitment.cu fr-tensor.cu g1-tensor.cu proof.cu zkrelu.cu zkfc.cu tlookup.cu polynomial.cu zksoftmax.cu rescaling.cu
CU_OBJS := $(CU_SRCS:.cu=.o)
CPP_SRCS := $(wildcard *.cpp)
CPP_OBJS := $(CPP_SRCS:.cpp=.o)
TARGETS := main ppgen commit-param self-attn ffn rmsnorm skip-connection
TARGET_OBJS := $(TARGETS:=.o)

# Pattern rule for CUDA source files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -dc $(NVCC_LTO_FLAGS) $< -o $@

# Pattern rule for C++ source files
%.o: %.cpp
	$(NVCC) -x cu $(NVCC_FLAGS) $(INCLUDES) -dc $(NVCC_LTO_FLAGS) $< -o $@

# General rule for building each target
$(TARGETS): % : %.o $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) $^ -o $@ $(NVCC_LTO_FLAGS)

# Clean rule
clean:
	rm -f $(TARGET_OBJS) $(CU_OBJS) $(CPP_OBJS) $(TARGETS)

# Default rule
all: $(TARGETS)