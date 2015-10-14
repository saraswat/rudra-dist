theanolearner: theanolearner.c
	echo "Building theanolearner.o"
	gcc -fPIC -c $$(python-config --cflags) -I$$(python -c 'import numpy; print(numpy.get_include())') -I../cpp/include theanolearner.c -o theanolearner.o
	gcc -shared $$(python-config --ldflags) -o theanolearner.so theanolearner.o

all: theanolearner

clean:
	$(RM) theanolearner.o
	$(RM) theanolearner.so

