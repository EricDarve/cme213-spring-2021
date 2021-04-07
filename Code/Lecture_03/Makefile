CPP := g++
CPPFLAGS := -std=c++11 -Wall
LDFLAGS := -lpthread

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
