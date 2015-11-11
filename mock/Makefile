install:    librudralearner-mock.so
	cp *.so $(RUDRA_LIB)

librudralearner-mock.so:    NativeLearner_Mock.cpp
	g++ -shared -fPIC -g NativeLearner_Mock.cpp -I$(RUDRA_INCLUDE) -o librudralearner-mock.so

clean:
	$(RM) *.so

