#include "commons.h"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include "db.hpp"
#include "rudra.pb.h"
#include <sys/time.h>
DEFINE_string(backend, "lmdb",
	      "The backend {leveldb, lmdb} containing the images"); // to use it --backend lmdb/leveldb
using namespace rudra;
using std::cout;
using std::endl;
struct timeval start;
struct timeval end;
float getTimeDelta(){
    gettimeofday(&end, NULL);
	time_t startSec = start.tv_sec;
	suseconds_t startUSec = start.tv_usec;
	time_t endSec = end.tv_sec;
	suseconds_t endUSec = end.tv_usec;
	return float(((endSec - startSec) * 1e6 + (endUSec - startUSec)) / 1e6);
}

int main(int argc, char** argv){
    gflags::SetUsageMessage("Decode an lmdb image table to Rudra binary format"
        "\n"
        "Usage:\n"
        "    decoder [FLAGS] INPUT_DB OUTPUT_DATA_FILE OUTPUT_LABEL_FILE LOG_FILE \n");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::cout<<"backend db "<<(FLAGS_backend)<<std::endl;
    // step1 open the database
    rudra::db::DB *db = (rudra::db::GetDB(FLAGS_backend));
    db->Open(argv[1], rudra::db::READ);
    //boost::scoped_ptr<rudra::db::Cursor> cursor(db->NewCursor());
    rudra::db::Cursor *cursor = (db->NewCursor());
    Datum datum;
    int count = 0;
    size_t img_size = 0;
    // step 1 get number of images and image size 
    while(cursor->valid()){
	count++;
	if(count == 1){
	    datum.ParseFromString(cursor->value());
	    img_size = datum.data().size();
	}
	// std::cout<<"cursor key: "<<cursor->key()<<std::endl;
	// cout<<"datum.data.size():"<<datum.data().size()<<endl;
	// int label = datum.label();
	// cout<<"datum label: "<<label<<endl;
	cursor->Next();
    }
    cout<<"count: "<<count<<endl;
    cout<<"image size:"<<img_size<<endl;

    gettimeofday(&start, NULL);
    // step2 open data , label, log file
    std::ofstream data_file(argv[2], std::ios::out | std::ios::trunc | std::ios::binary); 
    std::ofstream label_file(argv[3], std::ios::out | std::ios::trunc | std::ios::binary);
    std::ofstream log_file(argv[4], std::ios::out | std::ios::trunc);
    // step 2.1 write some metadata to data_file, label_file
    uint32_t count_be = htobe32(count);
    uint32_t img_size_be = htobe32(img_size);
    data_file.write((char*)&count_be, sizeof(uint32_t));
    data_file.write((char*)&img_size_be, sizeof(uint32_t));
    uint32_t label_dim_be = htobe32(1); // always just one-dimension for label;
    label_file.write((char*)&count_be, sizeof(uint32_t));
    label_file.write((char*)&label_dim_be, sizeof(uint32_t));
    cursor->SeekToFirst();
    int idx = 0;
    uint32_t label_bigendian = -1;
    while(cursor->valid()){
	
	datum.ParseFromString(cursor->value());
	data_file.write((datum.data().c_str()), img_size);
	uint32_t label_host = datum.label();
	#if __BYTE_ORDER == __LITTLE_ENDIAN
	// little endian code
	
	label_bigendian = htobe32(label_host);
        #else
	label_bigendian = label_host; // unchanged
        #endif
	label_file.write((char*)&label_bigendian, sizeof(uint32_t));
	if((++idx) % 1000 == 0){
	    std::cout<<"processed "<<idx<<"files"<<std::endl;
	    std::cout<<getTimeDelta()<<"seconds have passed"<<std::endl;
	}
	log_file<<idx<<"\t"<<cursor->key()<<"\t"<<label_host<<std::endl;
	cursor->Next();
	
    }
    log_file<<"processed in: "<<getTimeDelta()<<" seconds"<<std::endl;
    data_file.close();
    label_file.close();
    log_file.close();
    // std::cout<<"cursor key: "<<cursor->key()<<" cursor val size: "<<cursor->value().size()<<std::endl;
    // Datum datum;
    // datum.ParseFromString(cursor->value());
    // cout<<"datum.data.size():"<<datum.data().size()<<endl;
    // size_t datum_size = datum.data().size();
    // int label = datum.label();
    // cout<<"datum label: "<< label<<endl;
    // // open file to write 
    // std::ofstream f1(argv[2], std::ios::out | std::ios::trunc | std::ios::binary); 
    // f1.write((datum.data().c_str()), datum_size);
    // f1.close();
    // cout<<"datum.float_data_size():"<<datum.float_data_size()<<endl;
	//" cursor value: "<<cursor->value()<<std::endl;
}
