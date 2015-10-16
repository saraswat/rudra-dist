rudra-theano:
	mkdir -p include
	mkdir -p lib
	cd cpp && make
	cd theano && make
	cd x10 && make

clean:
	rm -rf ./lib ./include ./rudra
	cd cpp && make clean
	cd theano && make clean
	cd x10 && make clean

.PHONY: all clean
