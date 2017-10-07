CXX := g++
NVCC := $(shell which nvcc)
RM := rm -rf

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
	$(CXX) -o $@ -c $^ $(INCLUDES) $(CXXFLAGS)
	cp $@ $(OBJDIR)/$@ 

$(CUOBJECTS): %.o: %.cu
	$(NVCC) -o $@ -c $^ $(INCLUDES) $(NVCCFLAGS)
	echo $(NVCCFLAGS)
	cp $@ $(OBJDIR)/$@

.PHONY: clean
clean:
	$(RM) $(OBJECTS)
