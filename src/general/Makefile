CXX := g++
CXXFLAGS = -g -std=c++11 -O3 -Wall -m64 -Wextra
RM := rm -rf

ROOT := $(shell pwd)
SOURCES := $(wildcard *.cpp)
OBJECTS := $(SOURCES:.cpp=.o)
INCDIRS := $(ROOT)/../../inc

INCLUDES := -I$(INCDIRS)/general

CUDAINC := /usr/local/cuda-8.0/include
INCLUDES += -I$(CUDAINC)

all: $(OBJECTS)
$(OBJECTS): %.o: %.cpp
	$(CXX) -o $@ -c $^ $(INCLUDES) $(CXXFLAGS)

.PHONY: clean
clean:
	$(RM) $(OBJECTS)
