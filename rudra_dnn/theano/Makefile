theano: theano.c
	echo "Building theano.o"
	gcc -fPIC -c $$(python-config --cflags) -I$$(python -c 'import numpy; print(numpy.get_include())') -I../cpp/include theano.c -o theano.o
	gcc -shared $$(python-config --ldflags) -o theano.so theano.o

all: theano 

clean:
	$(RM) theano.o
	$(RM) theano.so

