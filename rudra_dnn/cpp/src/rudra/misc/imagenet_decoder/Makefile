SRC := decoder.cpp
OBJS = $(SRC:%.cpp=%.o)
DBSRC := db.cpp db_lmdb.cpp
DBOBJS = $(DBSRC:%.cpp=%.o)
PBSRC := rudra.pb.cc # proto buffer 
PBOBJS = $(PBSRC:%.cc=%.o)

GFLAGS_INCLUDE_PATH ?= /usr/include/
GFLAGS_LIB_PATH ?= /usr/lib64/
GLOG_INCLUDE_PATH ?= /usr/include/
GLOG_LIB_PATH ?=  /usr/lib64/
LMDB_INCLUDE_PATH ?= /usr/include
LMDB_LIB_PATH ?= /usr/lib64/
PROTOBUF_INCLUDE_PATH ?= /usr/include
PROTOBUF_LIB_PATH ?= /usr/lib64

CXXFLAGS += -I$(GFLAGS_INCLUDE_PATH) -I$(GLOG_INCLUDE_PATH) -I$(LMDB_INCLUDE_PATH) 
LDFLAGS  += -L$(GFLAGS_LIB_PATH) -lgflags -L$(GLOG_LIB_PATH) -lglog -L$(LMDB_LIB_PATH) -llmdb -L$(PROTOBUF_LIB_PATH) -lprotobuf
all: decoder

%.o:     %.cpp
	g++ $(CXXFLAGS) -c $< -o $@

%.o:     %.cc
	g++ $(CXXFLAGS) -c $< -o $@
rudra.pb.cc:	rudra.proto
	protoc -I=./ --cpp_out=./ rudra.proto
rudra.pb.o:	rudra.pb.cc
	g++ $(CXXFLAGS) -c rudra.pb.cc -o rudra.pb.o
decoder: decoder.cpp $(DBOBJS) $(PBOBJS)
	g++ $(CXXFLAGS) decoder.cpp $(DBOBJS) $(PBOBJS) $(LDFLAGS) -o "$@"

clean:
	-$(RM) decoder *.o 
