rudra-theano:
	mkdir -p include
	mkdir -p lib
	cd cpp && make
	cd theano && make
	cd x10 && make X10RTIMPL=sockets # use sockets until we sort out X10/MPI on DCC

clean:
	rm -rf ./lib ./include ./rudra
	cd cpp && make clean
	cd theano && make clean
	cd x10 && make clean

.PHONY: all clean
