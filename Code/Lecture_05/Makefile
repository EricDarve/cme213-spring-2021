CPP := g++

# macOS
CPPFLAGS := -std=c++11 -Wall -O2 -I/usr/local/include -Xpreprocessor -fopenmp -L/usr/local/lib -lomp

# Linux
# CPPFLAGS := -std=c++11 -Wall -O2 -fopenmp

SRC := $(wildcard *.cpp)
EXE := $(patsubst %.cpp,%,$(filter %.cpp,$(SRC)))

.PHONY: all
all: $(EXE)

# pattern rules
% : %.cpp
	$(CPP) $(CPPFLAGS) -o $@ $< $(LIB)

benchmark: matrix_prod_task
	for p in {1..4}; do ./matrix_prod_task -p $$p; done

.PHONY: clean
clean:
	$(RM) $(EXE)
