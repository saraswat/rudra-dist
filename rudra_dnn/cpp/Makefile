# Invocation:
#      make <target> OPT=<opt> ARCH=<architecture> PROFILE=yes
# Examples:
#	make OPT=-g ARCH=linux PROFILE=yes  # debuggable executable, g++ compiler,                     profiling information
#	make                                # optimized with -O3,    compiler determined from `uname`, no profiling information
#
#
# POZOR! ATTENTION! VORSICHT! XIAOXIN! KIKEN! SAWDHAN!
#
# To use a particular compiler, make sure that /usr/share/openmpi/mpic++-wrapper-data.txt
# specifies that compiler=....
# Merely setting CXX does not work

#vj fixed to remove rudra-learner targets and auxiliary definitions
RUDRA_LIB_SRC := src/rudra/MLPparams.cpp $(wildcard src/rudra/io/*.cpp) $(wildcard src/rudra/util/*.cpp)
RUDRA_LIB_OBJS =  $(RUDRA_LIB_SRC:%.cpp=%.o)
#echo "RUDRA_LIB_OBJS is $RUDRA_LIB_OBJS";

MPISRC := $(wildcard src/rudra/mpi/*.cpp) $(wildcard src/rudra/learner/*.cpp) $(wildcard src/rudra/param/*.cpp) $(wildcard src/rudra/sample/*.cpp) $(wildcard src/rudra/stat/*.cpp)
MPIOBJS = $(MPISRC:%.cpp=%.o)
PUBLICHEADERS :=  rudra/MLPparams.h rudra/io/SampleClient.h rudra/io/BinarySampleReader.h rudra/io/CharBinarySampleReader.h rudra/io/CharBinarySampleSeqReader.h rudra/io/BinarySampleSeqReader.h rudra/io/BinarySampleMFReader.h rudra/io/BinarySampleMFSeqReader.h rudra/io/GPFSSampleClient.h rudra/io/SampleReader.h rudra/io/UnifiedBinarySampleReader.h rudra/io/UnifiedBinarySampleSeqReader.h  rudra/util/*.h 
HEADERDIR := $(CURDIR)/include
LIBDIR := lib
RUDRA_LIB := $(LIBDIR)/librudra.a
#USE_GCC := YES

RUDRA_HOME ?= $(CURDIR)/..
# run ./install_hdf5.sh to build HDF5 under current directory
HDF5_INCLUDE_PATH ?= $(CURDIR)/hdf5/include
HDF5_LIB_PATH ?= $(CURDIR)/hdf5/lib
#OPENBLAS_LIB_PATH ?= /usr/lib64

# If no optimization parameter set then do optimize 
# as opposed to compile for debugging
OPT ?= -O3


# Use the architecture of host machine to determine what compiler to use.
# Set CXX and other environment variables expressing the choice.
# Unfortunately, mpicxx does not use CXX to determine its compiler;
# instead it uses the setitng inside mpic++-wrapper-data.txt.
# Therefore you can use the parameter ARCH to force architecture consistent 
# with mpic++-wrapper-data.txt.

ifdef ARCH
    arch=$(ARCH)
else
    arch=$(shell uname -p)
    ifeq ($(arch),unknown)
        arch=$(shell uname -m)
    endif
endif

ifeq ($(arch),ppc64)
    # POWER7 (inc. p775)
	CXX = xlC_r
    BLASLIB ?= ESSL
endif

ifeq ($(arch),ppc64le)
    # POWER8 LE
    CXX = xlC_r
endif

# compiler-specific flags
ifeq ($(CXX),xlC_r)
    CXXFLAGS += $(OPT) -q64 -qsmp
else
    # assume g++
    CXXFLAGS += -std=c++0x $(OPT) -w -Wno-strict-aliasing -fopenmp -DHAS_TR1_RANDOM

    ifneq (,$(findstring -g,$(OPT))) 
         # generate information for printing backtrace
         CXXFLAGS += -rdynamic   
    endif
endif


CXXFLAGS += -DNDEBUG -I$(RUDRA_HOME)/cpp/src -I$(HDF5_INCLUDE_PATH)
#CXXFLAGS += -DGEMM -I$(RUDRA_HOME)/cpp/src -I$(HDF5_INCLUDE_PATH)
LDFLAGS += -L$(HDF5_LIB_PATH) -lhdf5 -lhdf5_cpp


PROFILE ?=
ifdef PROFILE
	# enable CPU profiling with google-perftools
    LDFLAGS += -Wl,--no-as-needed -lprofiler -Wl,--as-needed
endif


# All Target
all: rudra_lib # rudra_devTest rudra_diff rudra_mpi_test

#src/rudra/mpi/%.o:	src/rudra/mpi/%.cpp
#	mpicxx $(CXXFLAGS) -c $< -o $@
src/rudra/param/%.o:	src/rudra/param/%.cpp
	mpicxx $(CXXFLAGS) -c $< -o $@
src/rudra/sample/%.o:	src/rudra/sample/%.cpp
	mpicxx $(CXXFLAGS) -c $< -o $@
src/rudra/stat/%.o:	src/rudra/stat/%.cpp
	mpicxx $(CXXFLAGS) -c $< -o $@
rudra_lib: $(RUDRA_LIB)

all: rudra_lib

$(RUDRA_LIB):	$(RUDRA_LIB_OBJS) $(LIBDIR) copy_headers
	$(AR) rs $@ $(RUDRA_LIB_OBJS)

copy_headers:	$(HEADERDIR)
#	cd src && cp --parents $(PUBLICHEADERS) $(HEADERDIR)/.
	rm -fr include
	mkdir include include/rudra include/rudra/io include/rudra/util
	cp src/rudra/MLPparams.h include/rudra
	cp src/rudra/io/*.h include/rudra/io/
	cp src/rudra/util/*.h include/rudra/util/

$(HEADERDIR):
	mkdir -p $(HEADERDIR)
$(LIBDIR):
	mkdir -p $(LIBDIR)

TESTSRC := $(wildcard test/rudra/*/*.cpp)
TESTS = $(TESTSRC:%.cpp=%)

% :	%.cpp
	$(CXX) $(CXXFLAGS)

test:	$(CNNLIB) $(TESTS)

clean:
#	-$(RM) rudra rudra_standalone rudra_inference rudra_devTest $(CNNOBJS) $(CUDAOBJS)  $(MPIOBJS); $(RM) -r $(HEADERDIR)/* $(LIBDIR)/*
	-$(RM) rudra rudra_devTest $(MPIOBJS); $(RM) -r $(HEADERDIR)/* $(LIBDIR)/*
	-$(RM) lib/librudra.a $(RUDRA_LIB_OBJS)
	-@echo ' '

.PHONY: all clean copy_headers
.SECONDARY:
