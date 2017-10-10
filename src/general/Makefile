CXX := g++
NVCC := $(shell which nvcc)
RM := rm -rf
CP := cp

ROOT := $(shell pwd)
SOURCES := $(wildcard *.cpp)
OBJECTS := $(SOURCES:.cpp=.o)
CUSOURCES := $(wildcard *.cu)
CUOBJECTS := $(CUSOURCES:.cu=.o)
INCDIRS := $(ROOT)/../../inc

INCLUDES := -I$(INCDIRS)/general

CUDAINC := /usr/local/cuda-8.0/include
INCLUDES += -I$(CUDAINC)

OBJDIR := $(ROOT)/../../obj

all: $(OBJECTS) $(CUOBJECTS)
$(OBJECTS): %.o: %.cpp
	$(CXX) -o $(OBJDIR)/$@ -c $^ $(INCLUDES) $(CXXFLAGS)

$(CUOBJECTS): %.o: %.cu
	$(NVCC) -o $(OBJDIR)/$@ -c $^ $(INCLUDES) $(NVCCFLAGS)

.PHONY: clean
clean:
	$(RM) $(OBJECTS)
