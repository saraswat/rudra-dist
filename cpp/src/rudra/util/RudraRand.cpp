/*
 * RudraRand.cpp
 *
 * Licensed Materials - Property of IBM
 *
 * Rudra Distributed Learning Platform
 *
 *  Copyright IBM Corp. 2016 All Rights Reserved
 */

#include "rudra/util/RudraRand.h"
#include <sys/time.h>
#include <iostream>
namespace rudra{
RudraRand::RudraRand(int rank, int threadid) : rank(rank), threadid(threadid) {
    struct timeval start;
    gettimeofday(&start, NULL);

    unsigned int seed = (unsigned int) ((float) start.tv_usec
		    / (float) ((rank + 1) * (rank + 1)));
    srand48_r(seed, &dd);
}

RudraRand::RudraRand(const RudraRand& rr){
    this->rank = rr.rank;
    this->threadid = rr.threadid;
    struct timeval start;
    gettimeofday(&start, NULL);

    unsigned int seed = (unsigned int) ((float) start.tv_usec
		    / (float) ((rank + 1) * (rank + 1)));
    srand48_r(seed, &dd);
}
RudraRand& RudraRand::operator=(const RudraRand& rhs){
if(&rhs != this){
    RudraRand(rhs.rank, rhs.threadid);
}
return *this;
}

long RudraRand::getLong(){
    long result;
    lrand48_r(&dd, &result);
    return result;
}

RudraRand::~RudraRand() { }

} /*namespace rudra*/
