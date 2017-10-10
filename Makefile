EXECUTABLE := my_spmv

EXCPDIRS := inc obj

SOURCES := my_spmv.cu

ROOT_DIR = $(shell pwd)

SUBDIRS := $(shell find . -maxdepth 1 -type d)

INCDIR = $(ROOT_DIR)/inc

SUBDIRS := $(basename $(patsubst ./%,%,$(SUBDIRS)))
SUBDIRS := $(filter-out $(EXCPDIRS),$(SUBDIRS))

INCLUDES := -I$(INCDIR)
INCLUDES += -I$(INCDIR)/general
INCLUDES += -I$(INCDIR)/matrix

CUDAINC = /usr/local/cuda-8.0/include

INCLUDES += -I$(CUDAINC)

OBJS = $(wildcard *.o)
OBJDIR = $(ROOT_DIR)/obj
OBJS += $(OBJDIR)/*.o

CXX := g++

CXXFLAGS = -g -std=c++11 -O3 -Wall -m64 -Wextra
export CXXFLAGS

NVCC = $(shell which nvcc)

RM = rm -rf

CP = cp

MAKE = make

CMD = ,
ifdef sm
	SM_ARCH = $(subst $(CMD),-,$(sm))
else
	SM_ARCH = 350
endif

ifeq (610, $(findstring 610, $(SM_ARCH)))
	SM = sm_61
endif
ifeq (520, $(findstring 520, $(SM_ARCH)))
    	SM = sm_52
endif
ifeq (370, $(findstring 370, $(SM_ARCH)))
    	SM = sm_37
endif
ifeq (350, $(findstring 350, $(SM_ARCH)))
    	SM = sm_35
endif
ifeq (300, $(findstring 300, $(SM_ARCH)))
   	SM = sm_30
endif

ifeq ($(verbose), 1)
        NVCCFLAGS += -v
endif

NVCCFLAGS = -std=c++11 -O3 -arch=$(SM) -Xcompiler -Wall -Xcompiler -Wextra -m64
export NVCCFLAGS

$(EXECUTABLE): my_spmv.o subdirs
	$(NVCC) -o $@ $(OBJS) $(INCLUDES) $(NVCCFLAGS)

subdirs: $(SUBDIRS)
	for dir in $^;\
	do $(MAKE) -C $$dir all||exit 1;\
	done

my_spmv.o: my_spmv.cu
	$(NVCC) -o $(OBJDIR)/$@ -c $^ $(INCLUDES) $(NVCCFLAGS) 

.PHONY: clean
clean:
	for dir in $(SUBDIRS);\
	do $(MAKE) -C $$dir clean||exit 1;\
	done
	$(RM) $(EXECUTABLE) $(OBJS)
