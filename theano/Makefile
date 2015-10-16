install:    librudralearner-theano.so theanolearner.so
	cp *.so $(RUDRA_LIB) 

theanolearner.so: theanolearner.c
	echo "Building theanolearner.o"
	gcc -fPIC -c $$(python-config --cflags) -I$$(python -c 'import numpy; print(numpy.get_include())') -I../cpp/include theanolearner.c -o theanolearner.o
	gcc -shared $$(python-config --ldflags) -o theanolearner.so theanolearner.o

librudralearner-theano.so:    NativeLearner_Theano.cpp
	g++ -shared -fPIC -g NativeLearner_Theano.cpp -I$(RUDRA_INCLUDE) -o librudralearner-theano.so

clean:
	$(RM) theanolearner.o
	$(RM) *.so

