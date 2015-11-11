rudra-theano:
	mkdir -p include
	mkdir -p lib
	cd cpp && make
	cd theano && make
	cd x10 && make X10RTIMPL=mpi RUDRA_LEARNER=rudralearner-theano

rudra-mock:
	mkdir -p include
	mkdir -p lib
	cd cpp && make
	cd mock && make
	cd x10 && make X10RTIMPL=mpi RUDRA_LEARNER=rudralearner-mock

clean:
	rm -rf ./lib ./include ./rudra
	cd cpp && make clean
	cd theano && make clean
	cd x10 && make clean

.PHONY: all clean
