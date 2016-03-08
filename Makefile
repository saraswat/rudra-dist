X10RTIMPL ?= mpi # [mpi | pami | sockets]

default: rudra-cudnn

rudra-theano:
	mkdir -p include
	mkdir -p lib
	cd cpp && make
	cd theano && make
	cd x10 && make X10RTIMPL=${X10RTIMPL} RUDRA_LEARNER=theano

# Rudra with IBM custom C++/OpenMP learner.
# See https://github.rtp.raleigh.ibm.com/rudra/rudra-learner
rudra-basic:
	mkdir -p include
	mkdir -p lib
	cd cpp && make
	cd x10 && make X10RTIMPL=${X10RTIMPL} RUDRA_LEARNER=basic

# Rudra with IBM cuDNN learner.
# See https://github.rtp.raleigh.ibm.com/rudra/rudra-cudnnlearner
rudra-cudnn:
	mkdir -p include
	mkdir -p lib
	cd cpp && make
	cd x10 && make X10RTIMPL=${X10RTIMPL} RUDRA_LEARNER=cudnn

rudra-mock:
	mkdir -p include
	mkdir -p lib
	cd cpp && make
	cd mock && make
	cd x10 && make X10RTIMPL=${X10RTIMPL} RUDRA_LEARNER=mock

clean:
	rm -rf ./lib ./include ./rudra-theano ./rudra-basic ./rudra-mock
	cd cpp && make clean
	cd theano && make clean
	cd x10 && make clean

.PHONY: all clean rudra-theano rudra-cudnn rudra-basic rudra-mock
