ifndef USE_GCC
    CXX = xlC_r
endif
CXXFLAGS += -I../utils -I../shared
ifeq ($(CXX),xlC_r)
    CXXFLAGS += -g -q64 -O3 -qsmp
    LDFLAGS += -q64 -L/usr/lib64 -lesslsmp6464 -L/opt/ibmcmp/xlf/14.1/lib64 -lxlf90_r -L/opt/ibmcmp/xlsmp/3.1/lib64/ -lxlfmath -lxl  -lxlsmp
else
    CXXFLAGS += -g -Wall -O2 -fopenmp
endif

OBJS := timer.o parser.o logger.o PerfTimer.o RudraRand.o

all:  $(OBJS)

%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	$(RM) $(OBJS)
	
