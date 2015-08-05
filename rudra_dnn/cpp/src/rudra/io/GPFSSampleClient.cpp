/**
 * GPFSSampleClient.cpp
 */

#include "rudra/MLPparams.h"
#include "rudra/io/GPFSSampleClient.h"
#include "rudra/util/MatrixContainer.h"
#include <cstring>
#include <pthread.h>

namespace rudra {
GPFSSampleClient::GPFSSampleClient(std::string name, bool threaded,
		SampleReader* reader) :
		clientName(name), threaded(threaded), sampleReader(reader), X(
				MatrixContainer<float>(MLPparams::_numInputDim, MLPparams::_batchSize)), Y(
				MatrixContainer<float>(MLPparams::_labelDim, MLPparams::_batchSize)), finishedFlag(
				false) {
	this->init();
}

void GPFSSampleClient::init(){
    this->count = 0;
    if(threaded){
	pthread_mutex_init(&(mutex), NULL);
	pthread_cond_init(&(fill), NULL);
	pthread_cond_init(&(empty), NULL);
	this->startProducerThd();
    }
    
}


struct p_thd_args{
    GPFSSampleClient *instance;
};

void GPFSSampleClient::startProducerThd(){
    p_thd_args *pta = new p_thd_args();
    pta->instance = this;
    pthread_create(&producerTID, NULL, &(GPFSSampleClient::producerThdHook), pta);
}

void *GPFSSampleClient::producerThdHook(void *args)
{
    p_thd_args *pargs = (p_thd_args*) args;
    pargs->instance->producerThdFunc(NULL);
    //delete pargs;// can afford this memory leak;
    return NULL;
}

void GPFSSampleClient:: producerThdFunc(void *args){
    while(!finishedFlag){
	pthread_mutex_lock(&mutex);
	while((count == GPFS_BUFFER_COUNT) && !finishedFlag){
	    pthread_cond_wait(&empty, &mutex);
	}
	// produce
	if(finishedFlag){
	    return;
	}
	sampleReader->readLabelledSamples(MLPparams::_batchSize, X, Y);
	count++; // don't forget to increment count
	pthread_cond_signal(&fill);
	pthread_mutex_unlock(&mutex);
    }

}

void GPFSSampleClient::getLabelledSamples(MatrixContainer<float> &samples,
		MatrixContainer<float> &labels) {
	if (threaded) {
		pthread_mutex_lock(&mutex);
		while (count == 0) {
			pthread_cond_wait(&fill, &mutex);
		}
		// consume()
		samples = X;
		labels = Y;
		//memcpy(dMat->buf, X.buf, sizePerImg * sizeof(float) * mbSize); // dont forget float takes 4 bytes and multiplies with minibatch size! April 28, 2015
		//memcpy(lMat->buf, Y.buf, sizePerLabel * sizeof(float) * mbSize);
		count--; // don't forget the decrement count
		pthread_cond_signal(&empty);
		pthread_mutex_unlock(&mutex);
	} else {
		sampleReader->readLabelledSamples(MLPparams::_batchSize, X, Y);
		samples = X;
		labels = Y;
		//memcpy(dMat->buf, X.buf, sizePerImg * sizeof(float) * mbSize); // dont forget float takes 4 bytes and multiplies with minibatch size! April 28, 2015
		//memcpy(lMat->buf, Y.buf, sizePerLabel * sizeof(float) * mbSize);
	}
}

GPFSSampleClient::~GPFSSampleClient() {
	if (threaded) {
		pthread_mutex_lock(&mutex);
		finishedFlag = true;
		pthread_cond_signal(&empty);
		pthread_mutex_unlock(&mutex);
		pthread_join(producerTID, NULL); // join the producer thread
	}
}
} /* namespace rudra */
