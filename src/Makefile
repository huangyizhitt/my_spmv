CXX := g++
NVCC := $(shell which nvcc)
RM := rm -rf

ROOT := $(shell pwd)
SOURCES := $(wildcard *.cpp)
OBJECTS := $(SOURCES:.cpp=.o)
CUSOURCES := $(wildcard *.cu)
CUOBJECTS := $(SOURCES:.cu=.o)
INCDIRS := $(ROOT)/../../inc
SUBDIRS := $(shell find . -maxdepth 1 -type d)
SUBDIRS := $(basename $(patsubst ./%,%,$(SUBDIRS)))

INCLUDES := -I$(INCDIRS)/general

CUDAINC := /usr/local/cuda-8.0/include
INCLUDES += -I$(CUDAINC)
OBJDIR := $(ROOT)/../../obj

all: $(OBJECTS) $(CUOBJECTS) subdirs
subdirs: $(SUBDIRS)
	for dir in $^;\
	do $(MAKE) -C $$dir all||exit 1;\
	done

$(OBJECTS): %.o: %.cpp
	$(CXX) -o $@ -c $^ $(INCLUDES) $(CXXFLAGS)
	cp $@ $(OBJDIR)/$@

$(CUOBJECTS): %.o: %.cu
	$(NVCC) -o $@ -c $^ $(INCLUDES) $(NVCCFLAGS)
	cp $@ $(OBJDIR)/$@

.PHONY: clean
clean:
	for dir in $(SUBDIRS);\
	do $(MAKE) -C $$dir clean||exit 1;\
	done
	$(RM) $(OBJECTS)
