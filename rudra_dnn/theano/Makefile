theano: theano.c theano.h
	echo "Building theano.o"
#	gcc  -c $$(/usr/bin/python2.7-config --cflags) theano.c $$(/usr/bin/python2.7-config --ldflags) -o theano.o
# TODO fix this to get the right include path for anaconda automatically
	gcc -c $$(python-config --cflags) -I/Users/vijaysaraswat/anaconda/lib/python2.7/site-packages/numpy/core/include/ theano.c $$(python-config --ldflags) -o theano.o
	ar rs libtheano.a theano.o
	ranlib libtheano.a
	#gcc -shared -Wl,-soname,libtheano.so.1 -o libtheano.so.1.0.1  theano.o
all: theano 

clean:
	$(RM) theano.o
	$(RM) libtheano.a

