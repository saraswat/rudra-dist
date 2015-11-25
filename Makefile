# set X10_TRANSPORT to mpi or sockets
X10_TRANSPORT ?= sockets
rudra-theano:
	mkdir -p include
	mkdir -p lib
	cd cpp && make
	cd theano && make
	cd x10 && make X10RTIMPL=${X10_TRANSPORT} RUDRA_LEARNER=rudralearner-theano

# Rudra with IBM custom C++/OpenMP learner.
# See https://github.rtp.raleigh.ibm.com/rudra/rudra-learner
rudra-basic:    lib/librudralearner-basic.so
	mkdir -p include
	mkdir -p lib
	cd cpp && make
	cd x10 && make X10RTIMPL=${X10_TRANSPORT} RUDRA_LEARNER=rudralearner-basic

rudra-mock:
	mkdir -p include
	mkdir -p lib
	cd cpp && make
	cd mock && make
	cd x10 && make X10RTIMPL=${X10_TRANSPORT} RUDRA_LEARNER=rudralearner-mock

clean:
	rm -rf ./lib ./include ./rudra
	cd cpp && make clean
	cd theano && make clean
	cd x10 && make clean

.PHONY: all clean
