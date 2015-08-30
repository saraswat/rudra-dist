arch=$(shell uname -p)
ifeq ($(arch),unknown)
    arch=$(shell uname -m)
endif

X10CXX_PREARGS += -VERBOSE_CHECKS

ifeq ($(arch),ppc64)
    # POWER7 (inc. p775)
	CXX = xlC_r
    BLASLIB ?= ESSL
    X10CXX := $(X10_HOME)/x10.dist/bin/x10c++ -x10rt pami
else
    X10CXX := $(X10_HOME)/x10.dist/bin/x10c++ #-x10rt mpi
    X10CXX_PREARGS += -cxx-prearg -std=c++0x -cxx-prearg -DHAS_TR1_RANDOM -cxx-prearg -fopenmp
endif

# enable CPU profiling with google-perftools
PROFILE ?=
ifdef PROFILE
  X10CXX_PREARGS += -gpt
endif

# X10CXX_PREARGS = -cxx-prearg -I/gpfs/my_gpfs/home/weiwisc/cnn_tidy_git/cpp/install/include 
# X10CXX_POSTARGS = -cxx-postarg -L/usr/lib64 -cxx-postarg -lesslsmp6464 -cxx-postarg -L/opt/ibmcmp/xlf/14.1/lib64 -cxx-postarg -lxlf90_r  -cxx-postarg -L/opt/ibmcmp/xlsmp/3.1/lib64/ -cxx-postarg -lxlsmp -cxx-postarg -lxlfmath -cxx-postarg -lxl -cxx-postarg -L/gpfs/my_gpfs/home/weiwisc/cnn_tidy_git/cpp/install/lib -cxx-postarg -lcnn 

# TODO: Arnaud -- this needs to be set up properly to work for everyone. 
# We could import the scripts from Rudra to install hdf5
export HDF5_INCLUDE_PATH=/Users/vijaysaraswat/code/rudra-dist2/rudra_dnn/cpp/hdf5/include
export HDF5_LIB_PATH=/Users/vijaysaraswat/code/rudra-dist2/rudra_dnn/cpp/hdf5/lib

X10CXX_PREARGS += -cxx-prearg -I${RUDRA_HOME}/cpp/include -cxx-prearg -I${HDF5_INCLUDE_PATH} -cxx-prearg -I${RUDRA_HOME}/theano #-I/usr/lib/openmpi/include 
X10CXX_POSTARGS += -cxx-postarg -L${RUDRA_HOME}/cpp/lib -cxx-postarg -lrudra -cxx-postarg -L$(HDF5_LIB_PATH) -cxx-postarg -lhdf5 -cxx-postarg -lhdf5_cpp -cxx-postarg -L${RUDRA_HOME}/theano  -cxx-postarg -ltheano

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
# vj need to add these for theano linkage. 
# TODO: Do this in a system generic way, this is obtained by 
# by splicing -cxx-postarg in between the terms returned by python2.7-conifig --ldflags
# and quoting
# -lpython2.7 -ldl -framework CoreFoundation -u _PyMac_Error
#X10CXX_POSTARGS += -cxx-postarg -L/usr/lib/python2.7/config-x86_64-linux-gnu -cxx-postarg -L/usr/lib -cxx-postarg -lpthread -cxx-postarg -ldl  -cxx-postarg -lutil -cxx-postarg -lm  -cxx-postarg -lpython2.7 -cxx-postarg -Xlinker -cxx-postarg -export-dynamic -cxx-postarg "-Wl,-O1" -cxx-postarg "-Wl,-Bsymbolic-functions"
X10CXX_POSTARGS += -cxx-postarg -lpython2.7 -cxx-postarg -ldl -cxx-postarg -framework -cxx-postarg CoreFoundation -cxx-postarg -u -cxx-postarg _PyMac_Error 
X10CXX_POSTARTS += -cxx-postarg -L/usr/lib -cxx-postarg -lpthread -cxx-postarg -ldl  -cxx-postarg -lutil -cxx-postarg -lm  -cxx-postarg -lpython2.7 -cxx-postarg -Xlinker -cxx-postarg  -cxx-postarg "-Wl,-O1" -cxx-postarg "-Wl,-Bsymbolic-functions"
# -export-dynamic

all: rudra
rudra: src/xrudra/Rudra.x10 src/xrudra/Learner.x10 src/xrudra/Tester.x10 src/xrudra/ImmedLearner.x10 src/xrudra/ImmedReconciler.x10 src/xrudra/ApplyLearner.x10 src/xrudra/ApplyReconciler.x10 src/xrudra/HardSync.x10 src/xrudra/AtLeastRAllReducer.x10 src/xrudra/NativeLearner.cpp src/xrudra/NativeLearner.x10  src/xrudra/NativeLearner.h src/xrudra/util/*SwapBuffer.x10 src/xrudra/util/Timer.x10 src/xrudra/util/Logger.x10 
	$(X10CXX) -O -g -NO_CHECKS -sourcepath src -report postcompile=1 $(X10CXX_PREARGS) $(X10CXX_POSTARGS)  -d ./tmp -o ./rudra src/xrudra/Rudra.x10

clean:
	$(RM) rudra
	$(RM) -fr ./tmp
