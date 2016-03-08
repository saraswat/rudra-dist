/*
 * RudraRand.h
 *
 * Licensed Materials - Property of IBM
 *
 * Rudra Distributed Learning Platform
 *
 *  Copyright IBM Corp. 2016 All Rights Reserved
 */

#ifndef __RUDRA_UTIL_RAND_H_
#define __RUDRA_UTIL_RAND_H_
#include <cstdlib>
// a thread-safe random number generator
namespace rudra{
    class RudraRand{
    public:
	// vj TODO: conditionalize this to use nrand48 for macos and drand48 where it exists
	drand48_data dd;
	//unsigned short  dd[3];
	RudraRand& operator=(const RudraRand& rhs);// overload assignment
	int rank;
	int threadid;
	long getLong();
        RudraRand() {}
	RudraRand(int rank, int threadid);
	RudraRand(const RudraRand& rr);// copy constrcutor
	
  
	~RudraRand();
    }; // end of class

}

#endif
