# ==== START CONFIGURATION ====

# All of these parameters may be set on the command line or in this file.

# Full path to x10c++ if not in $PATH
X10CXX_BIN ?= x10c++

# Set to 1 to enable profiling
PROFILE ?=

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
    X10RTIMPL ?= pami
else
    X10CXX_PREARGS += -cxx-prearg -std=c++0x
    X10RTIMPL ?= mpi
endif

ifdef X10RTIMPL # sockets, mpi, pami, etc.
  X10FLAG += -x10rt ${X10RTIMPL}
endif

# enable CPU profiling with google-perftools
ifdef PROFILE
  X10CXX_PREARGS += -gpt
endif

X10CXX_PREARGS += -cxx-prearg -I${RUDRA_HOME}/cpp/include
X10CXX_POSTARGS += -cxx-postarg -L${RUDRA_HOME}/cpp/lib -cxx-postarg -lrudra

# vj need to add these for theano linkage.
# TODO: Do this in a system generic way, this is obtained by
# by splicing -cxx-postarg in between the terms returned by python2.7-config --ldflags
# and quoting
X10CXX_POSTARGS += -cxx-postarg -L/usr/lib/python2.7/config-x86_64-linux-gnu -cxx-postarg -L/usr/lib -cxx-postarg -lpthread -cxx-postarg -ldl  -cxx-postarg -lutil -cxx-postarg -lm  -cxx-postarg -lpython2.7 -cxx-postarg -Xlinker -cxx-postarg -export-dynamic -cxx-postarg "-Wl,-O1" -cxx-postarg "-Wl,-Bsymbolic-functions"

all: rudra
rudra: src/xrudra/Rudra.x10 src/xrudra/Learner.x10 src/xrudra/Tester.x10 src/xrudra/ImmedLearner.x10 src/xrudra/ImmedReconciler.x10 src/xrudra/ApplyLearner.x10 src/xrudra/ApplyReconciler.x10 src/xrudra/HardSync.x10 src/xrudra/AtLeastRAllReducer.x10 src/xrudra/NativeLearner.cpp src/xrudra/NativeLearner.x10  src/xrudra/NativeLearner.h src/xrudra/util/*SwapBuffer.x10 src/xrudra/util/Timer.x10 src/xrudra/util/Logger.x10 
	$(X10CXX) -O -g -NO_CHECKS -sourcepath src -report postcompile=1 $(X10CXX_PREARGS) $(X10CXX_POSTARGS) -d ./tmp -o ./rudra src/xrudra/Rudra.x10

clean:
	$(RM) rudra
	$(RM) -fr ./tmp
