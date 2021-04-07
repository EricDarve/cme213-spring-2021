CPP := g++

# macOS options
# CPPFLAGS := -std=c++11 -Wall -O2 -I/usr/local/include -Xpreprocessor -fopenmp
# LDFLAGS := -L/usr/local/lib -lomp

# g++ options for Linux
CPPFLAGS := -std=c++11 -Wall -O2 -fopenmp
LDFLAGS :=

SRC := $(wildcard *.cpp)
EXE := $(patsubst %.cpp,%,$(filter %.cpp,$(SRC)))

.PHONY: all
all: $(EXE)

# pattern rules
% : %.cpp
	$(CPP) $(CPPFLAGS) -o $@ $< $(LDFLAGS)

.PHONY: clean
clean:
	$(RM) $(EXE)
