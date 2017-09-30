EXECUTABLE := my_spmv

SOURCES := my_spmv.cu

ROOT_DIR = $(shell pwd)

SUBDIRS := $(shell find . -maxdepth 1 -type d)

INCDIR = $(ROOT_DIR)/inc

SUBDIRS := $(basename $(patsubst ./%,%,$(SUBDIRS)))
SUBDIRS := $(filter-out inc,$(SUBDIRS))

INCLUDES := -I$(INCDIR)
INCLUDES += -I$(INCDIR)/general

CUDAINC = /usr/local/cuda-8.0/include

INCLUDES += -I$(CUDAINC)

OBJ = $(wildcard $(SRCDIRS)/*.o)

CXX := g++

CXXFLAGS = -g -std=c++11 -O3 -Wall -m64 -Wextra

NVCC = "$(shell which nvcc)"

NVCCFLAGS = -O3 -Xcompiler -Wall -Xcompiler -Wextra -m64 -Xcompiler -ffloat-store

RM = rm -rf

MAKE = make

CMD = ,
ifdef sm
	SM_ARCH = $(subst $(CMD),-,$(sm))
else
	SM_ARCH = 350
endif

ifeq (610, $(findstring 610, $(SM_ARCH)))
	SM_TARGETS += -gencode=arch=compute_61, code=\"sm_61,compute_61\"
endif
ifeq (520, $(findstring 520, $(SM_ARCH)))
    SM_TARGETS  += -gencode=arch=compute_52,code=\"sm_52,compute_52\"
endif
ifeq (370, $(findstring 370, $(SM_ARCH)))
    SM_TARGETS  += -gencode=arch=compute_37,code=\"sm_37,compute_37\"
endif
ifeq (350, $(findstring 350, $(SM_ARCH)))
    SM_TARGETS  += -gencode=arch=compute_35,code=\"sm_35,compute_35\"
endif
ifeq (300, $(findstring 300, $(SM_ARCH)))
    SM_TARGETS  += -gencode=arch=compute_30,code=\"sm_30,compute_30\"
endif

ifeq ($(verbose), 1)
        NVCCFLAGS += -v
endif

$(EXECUTABLE): my_spmv.o subdirs
	$(NVCC) $(SM_TARGETS) -o $@ -c $^ $(INCLUDE) $(NVCCFLAGS)

subdirs: $(SUBDIRS)
	echo $(SUBDIRS)
	for dir in $^;\
	do $(MAKE) -C $$dir all||exit 1;\
	done

my_spmv.o: my_spmv.cu
	$(NVCC) $(SM_TARGETS) -o $@ -c $^ $(INCLUDES) $(NVCCFLAGS) 

.PHONY: clean
clean:
	for dir in $(SUBDIRS);\
	do $(MAKE) -C $$dir clean||exit 1;\
	done
	$(RM) $(EXECUTABLE) $(OBJ)
