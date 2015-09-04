# ==== START CONFIGURATION ====

# All of these parameters may be set on the command line or in this file.

# Full path to x10c++ if not in $PATH
X10CXX_BIN ?= x10c++

# Path to hdf5 include files / library if not standard
HDF5_INCLUDE_PATH ?=
HDF5_LIB_PATH ?=

# Set to 1 to enable profiling
PROFILE ?=

# Set to one of ESSL, OpenBLAS or leave blank
# for Netlib BLAS (links with -lblas)
BLASLIB ?=

# Set to OpenBLAS include / library path if not standard
# (only needed if BLASLIB=OpenBLAS)
OPENBLAS_INCLUDE_PATH ?=
OPENBLAS_LIB_PATH ?=

# ===== END CONFIGURATION =====

arch=$(shell uname -p)
ifeq ($(arch),unknown)
    arch=$(shell uname -m)
endif

X10CXX_PREARGS += -VERBOSE_CHECKS

ifeq ($(arch),ppc64)
    # POWER7 (inc. p775)
	CXX = xlC_r
    BLASLIB ?= ESSL
    X10CXX := $(X10CXX_BIN) -x10rt pami
else
    X10CXX := $(X10CXX_BIN) #-x10rt mpi
    X10CXX_PREARGS += -cxx-prearg -std=c++0x -cxx-prearg -DHAS_TR1_RANDOM -cxx-prearg -fopenmp
endif

# enable CPU profiling with google-perftools
ifdef PROFILE
  X10CXX_PREARGS += -gpt
endif

# X10CXX_POSTARGS = -cxx-postarg -L/usr/lib64 -cxx-postarg -lesslsmp6464 -cxx-postarg -L/opt/ibmcmp/xlf/14.1/lib64 -cxx-postarg -lxlf90_r  -cxx-postarg -L/opt/ibmcmp/xlsmp/3.1/lib64/ -cxx-postarg -lxlsmp -cxx-postarg -lxlfmath -cxx-postarg -lxl -cxx-postarg -L/gpfs/my_gpfs/home/weiwisc/cnn_tidy_git/cpp/install/lib -cxx-postarg -lcnn 

X10CXX_PREARGS += -cxx-prearg -I${RUDRA_HOME}/cpp/include -cxx-prearg -I${HDF5_INCLUDE_PATH}
X10CXX_POSTARGS += -cxx-postarg -L${RUDRA_HOME}/cpp/lib -cxx-postarg -lrudra -cxx-postarg -L$(HDF5_LIB_PATH) -cxx-postarg -lhdf5 -cxx-postarg -lhdf5_cpp

ifeq ($(BLASLIB),ESSL)
    # p775, power8
    X10CXX_PREARGS += -cxx-prearg -D__essl__
    X10CXX_POSTARGS += -cxx-postarg -D__essl__ -cxx-postarg -L/usr/lib64 -cxx-postarg -lesslsmp6464 -cxx-postarg -L/opt/ibmcmp/xlf/14.1/lib64 -cxx-postarg -lxlf90_r  -cxx-postarg -L/opt/ibmcmp/xlsmp/3.1/lib64/ -cxx-postarg -lxlsmp -cxx-postarg -lxlfmath -cxx-postarg -lxl  
else
    ifeq ($(BLASLIB),OpenBLAS)
        X10CXX_PREARGS += -cxx-prearg -I$(OPENBLAS_INCLUDE_PATH)
        X10CXX_POSTARGS += -cxx-postarg -L$(OPENBLAS_LIB_PATH) -cxx-postarg -lopenblas
    else
        # assume Netlib BLAS
        X10CXX_POSTARGS += -cxx-postarg -lblas
    endif
endif

all: rudra
rudra: src/xrudra/Rudra.x10 src/xrudra/Learner.x10 src/xrudra/Tester.x10 src/xrudra/ImmedLearner.x10 src/xrudra/ImmedReconciler.x10 src/xrudra/ApplyLearner.x10 src/xrudra/ApplyReconciler.x10 src/xrudra/HardSync.x10 src/xrudra/AtLeastRAllReducer.x10 src/xrudra/NativeLearner.cpp src/xrudra/NativeLearner.x10  src/xrudra/NativeLearner.h src/xrudra/util/*SwapBuffer.x10 src/xrudra/util/Timer.x10 src/xrudra/util/Logger.x10 
	$(X10CXX) -O -g -NO_CHECKS -sourcepath src -report postcompile=1 $(X10CXX_PREARGS) $(X10CXX_POSTARGS) -d ./tmp -o ./rudra src/xrudra/Rudra.x10

clean:
	$(RM) rudra
	$(RM) -fr ./tmp
