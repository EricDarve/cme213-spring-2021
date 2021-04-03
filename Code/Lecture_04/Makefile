# CPP := /usr/local/bin/g++-9
# CPPFLAGS := -std=c++11 -Wall -O2 -fopenmp
CPP := g++
CPPFLAGS := -std=c++11 -Wall -O2 -I/usr/local/include -Xpreprocessor -fopenmp
LDFLAGS := -L/usr/local/lib -lomp

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
