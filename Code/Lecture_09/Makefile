CXX=nvcc
CXXFLAGS=-O3 -Icuda_helper --gpu-architecture=compute_75 --gpu-code=sm_75

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
