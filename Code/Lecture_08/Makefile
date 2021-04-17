CXX=nvcc
CXXFLAGS=-Icuda_helper -O3 --gpu-architecture=compute_75 --gpu-code=sm_75
# CXXFLAGS=-Icuda_helper -O3
# CXXFLAGS=-Icuda_helper --gpu-architecture=compute_50 --gpu-code=sm_50,sm_75
# CXXFLAGS=-Icuda_helper --gpu-architecture=compute_50 --gpu-code=sm_50
# CXXFLAGS=-Icuda_helper --gpu-architecture=compute_50 --gpu-code=sm_75
# CXXFLAGS=-Icuda_helper --gpu-architecture=compute_75 --gpu-code=sm_50
# CXXFLAGS=-Icuda_helper --gpu-architecture=compute_50 --gpu-code=compute_50,sm_50
# CXXFLAGS=-Icuda_helper --gpu-architecture=compute_75 --gpu-code=sm_75
# CXXFLAGS=-Icuda_helper --gpu-architecture=compute_75
# CXXFLAGS=-Icuda_helper --gpu-architecture=sm_75
CXXFLAGS=-Icuda_helper -arch=sm_75

SRC := $(wildcard *.cpp) $(wildcard *.cu)
EXE := $(patsubst %.cpp,%,$(filter %.cpp,$(SRC))) $(patsubst %.cu,%,$(filter %.cu,$(SRC)))

.PHONY: all
all: $(EXE)

# pattern rules
% : %.cu
	$(CXX) $(CXXFLAGS) -o $@ $<

% : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

.PHONY: clean
clean:
	$(RM) $(EXE)
