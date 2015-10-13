# Invocation:
#      make <target> OPT=<opt> ARCH=<architecture> PROFILE=yes
# Examples:
#	make OPT=-g ARCH=linux PROFILE=yes  # debuggable executable, g++ compiler,                     profiling information
#	make                                # optimized with -O3,    compiler determined from `uname`, no profiling information
#
#

#vj fixed to remove rudra-learner targets and auxiliary definitions
RUDRA_LIB_SRC := src/rudra/MLPparams.cpp $(wildcard src/rudra/io/*.cpp) $(wildcard src/rudra/util/*.cpp)
RUDRA_LIB_OBJS =  $(RUDRA_LIB_SRC:%.cpp=%.o)
#echo "RUDRA_LIB_OBJS is $RUDRA_LIB_OBJS";

HEADERDIR := $(CURDIR)/include
LIBDIR := lib
RUDRA_LIB := $(LIBDIR)/librudra.a

RUDRA_HOME ?= $(CURDIR)/..

# If no optimization parameter set then do optimize
# as opposed to compile for debugging
OPT ?= -O3


# Use the architecture of host machine to determine what compiler to use.
# Set CXX and other environment variables expressing the choice.

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
    CXXFLAGS += -std=c++0x $(OPT) -w -Wno-strict-aliasing

    ifneq (,$(findstring -g,$(OPT)))
         # generate information for printing backtrace
         CXXFLAGS += -rdynamic
    endif
endif


CXXFLAGS += -DNDEBUG -I$(RUDRA_HOME)/cpp/src

PROFILE ?=
ifdef PROFILE
	# enable CPU profiling with google-perftools
    LDFLAGS += -Wl,--no-as-needed -lprofiler -Wl,--as-needed
endif


# All Target
all: rudra_lib

rudra_lib: $(RUDRA_LIB)

$(RUDRA_LIB):	$(RUDRA_LIB_OBJS) $(LIBDIR) copy_headers
	$(AR) rs $@ $(RUDRA_LIB_OBJS)

copy_headers:	$(HEADERDIR)
	rm -fr $(HEADERDIR)
	mkdir $(HEADERDIR) $(HEADERDIR)/rudra $(HEADERDIR)/rudra/io $(HEADERDIR)/rudra/util
	cp src/rudra/*.h $(HEADERDIR)/rudra
	cp src/rudra/io/*.h $(HEADERDIR)/rudra/io/
	cp src/rudra/util/*.h $(HEADERDIR)/rudra/util/

$(HEADERDIR):
	mkdir -p $(HEADERDIR)
$(LIBDIR):
	mkdir -p $(LIBDIR)

TESTSRC := $(wildcard test/rudra/*/*.cpp)
TESTS = $(TESTSRC:%.cpp=%)

% :	%.cpp
	$(CXX) $(CXXFLAGS)

test:	$(RUDRA_LIB) $(TESTS)

clean:
	-$(RM) -r $(HEADERDIR)/* $(LIBDIR)/*
	-$(RM) lib/librudra.a $(RUDRA_LIB_OBJS)
	-@echo ' '

.PHONY: all clean copy_headers
.SECONDARY:
