#include "rudra/util/RudraRand.h"
#include <sys/time.h>
namespace rudra{
  RudraRand::RudraRand(int rank, int threadid) : rank(rank), threadid(threadid) {
        dd[0] = 1; dd[1] = 2; dd[2] = 3;
        struct timeval start;
	gettimeofday(&start, NULL);

	unsigned int seed = (unsigned int) ((float) start.tv_usec
			/ (float) ((rank + 1) * (rank + 1)));
	//unsigned int seed = 12345;
	//vj	srand48_r(seed, &dd);
	// for macos using nrand
	//	dd = { 1, 2, 3};
    }
    RudraRand::RudraRand(const RudraRand& rr){
	this->rank = rr.rank;
	this->threadid = rr.threadid;
	this->dd[0] = rr.dd[0]; 	this->dd[1] = rr.dd[1]; 	this->dd[2] = rr.dd[2];
    }
    RudraRand& RudraRand::operator=(const RudraRand& rhs){
	if(&rhs != this){
	    RudraRand(rhs.rank, rhs.threadid);
	}
	return *this;
    }

    long RudraRand::getLong(){
	 long result;
	 //lrand48_r(&dd, &result);
         result = nrand48(dd);
	 return result;
    }
    RudraRand::~RudraRand(){
    }
    
}/*namespace rudra*/
