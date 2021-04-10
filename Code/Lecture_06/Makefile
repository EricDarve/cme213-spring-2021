CPP := g++

# macOS
CPPFLAGS := -DNDEBUG -std=c++11 -Wall -O2 -I/usr/local/include -Xpreprocessor -fopenmp -L/usr/local/lib -lomp

# Linux
# CPPFLAGS := -DNDEBUG -std=c++11 -Wall -O2 -fopenmp

SRC := $(wildcard *.cpp)
EXE := $(patsubst %.cpp,%,$(filter %.cpp,$(SRC)))

.PHONY: all
all: $(EXE)

# pattern rules
% : %.cpp
	$(CPP) $(CPPFLAGS) -o $@ $< $(LIB)

.PHONY: clean
clean:
	$(RM) $(EXE)
