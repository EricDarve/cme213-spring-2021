CXX=mpic++
CXXFLAGS=

SRC := $(wildcard *.cpp)
EXE := $(patsubst %.cpp,%,$(filter %.cpp,$(SRC)))

.PHONY: all
all: $(EXE)

# pattern rules
% : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

.PHONY: clean
clean:
	$(RM) $(EXE)


# Interactive commands
# salloc -N 4 -n 24
# srun --pty bash
